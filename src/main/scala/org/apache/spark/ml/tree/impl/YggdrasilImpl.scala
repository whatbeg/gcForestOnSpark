/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.tree.impl

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.tree.impl.YggdrasilUtil._
import org.apache.spark.ml.tree._
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.impurity.ImpurityCalculator
import org.apache.spark.mllib.tree.model.{ImpurityStats, Predict}
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.collection.BitSet
import org.roaringbitmap.RoaringBitmap


/**
  * DecisionTree which partitions data by feature.
  *
  * Algorithm:
  *  - Repartition data, grouping by feature.
  *  - Prep data (sort continuous features).
  *  - On each partition, initialize instance--node map with each instance at root node.
  *  - Iterate, training 1 new level of the tree at a time:
  *     - On each partition, for each feature on the partition, select the best split for each node.
  *     - Aggregate best split for each node.
  *     - Aggregate bit vector (1 bit/instance) indicating whether each instance splits
  *       left or right.
  *     - Broadcast bit vector.  On each partition, update instance--node map.
  *
  * TODO: Update to use a sparse column store.
  */
private[spark] object YggdrasilImpl extends Logging{

  def train(
       input: RDD[LabeledPoint],
       strategy: Strategy,
       colStoreInput: Option[RDD[(Int, Array[Double])]],
       parentUID: Option[String] = None): DecisionTreeModel = {
    val metadata = YggdrasilMetadata.fromStrategy(strategy)
    val numFeatures = input.first().features.size
    // The case with 1 node (depth = 0) is handled separately.
    // This allows all iterations in the depth > 0 case to use the same code.
    // TODO: Check that learning works when maxDepth > 0 but learning stops at 1 node (because of
    //       other parameters).
    if (strategy.maxDepth == 0) {
      val impurityAggregator: ImpurityAggregatorSingle =
        input.aggregate(metadata.createImpurityAggregator())(
          (agg, lp) => agg.update(lp.label, 1.0),
          (agg1, agg2) => agg1.add(agg2))
      val impurityCalculator = impurityAggregator.getCalculator
      val rootNode = new LeafNode(getPredict(impurityCalculator).predict, impurityCalculator.calculate(),
        impurityCalculator)
      return finalizeTree(rootNode, strategy.algo, strategy.numClasses, numFeatures,
        parentUID)
    }

    val timer = new TimeTracker()
    // Prepare column store.
    //   Note: rowToColumnStoreDense checks to make sure numRows < Int.MaxValue.
    // TODO: Is this mapping from arrays to iterators to arrays (when constructing learningData)?
    //       Or is the mapping implicit (i.e., not costly)?
    timer.start("rowToColumnStoreDense")
    val colStoreInit: RDD[(Int, Array[Double])] = colStoreInput.getOrElse(
      rowToColumnStoreDense(input.map(_.features)))
    val numRows: Int = colStoreInit.first()._2.length
    timer.stop("rowToColumnStoreDense")
    timer.start("training")
    val rootNode = if (metadata.numClasses > 1 && metadata.numClasses <= 32) {
        YggdrasilClassification.run(input, colStoreInit, metadata, numRows, strategy.maxDepth)
      } else {
        YggdrasilRegression.run(input, colStoreInit, metadata, numRows, strategy.maxDepth)
      }
    timer.stop("training")
    println("Internal timing for Decision Tree:")
    println(s"$timer")
    finalizeTree(rootNode, strategy.algo, strategy.numClasses, numFeatures,
      parentUID)
  }

  private[impl] def finalizeTree(
      rootNode: Node,
      algo: OldAlgo.Algo,
      numClasses: Int,
      numFeatures: Int,
      parentUID: Option[String]): DecisionTreeModel = {
    parentUID match {
      case Some(uid: String) =>
        if (algo == OldAlgo.Classification) {
          new DecisionTreeClassificationModel(uid, rootNode, numFeatures = numFeatures,
            numClasses = numClasses)
        } else {
          new DecisionTreeRegressionModel(uid, rootNode, numFeatures = numFeatures)
        }
      case None =>
        if (algo == OldAlgo.Classification) {
          new DecisionTreeClassificationModel(rootNode, numFeatures = numFeatures,
            numClasses = numClasses)
        } else {
          new DecisionTreeRegressionModel(rootNode, numFeatures = numFeatures)
        }
    }
  }

  private[impl] def getPredict(impurityCalculator: ImpurityCalculator): Predict = {
    val pred = impurityCalculator.predict
    new Predict(predict = pred, prob = impurityCalculator.prob(pred))
  }

  /**
    * On driver: Grow tree based on chosen splits, and compute new set of active nodes.
    * @param oldPeriphery  Old periphery of active nodes.
    * @param bestSplitsAndGains  Best (split, gain) pairs, which can be zipped with the old
    *                            periphery.  These stats will be used to replace the stats in
    *                            any nodes which are split.
    * @param minInfoGain  Threshold for min info gain required to split a node.
    * @return  New active node periphery.
    *          If a node is split, then this method will update its fields.
    */
  private[impl] def computeActiveNodePeriphery(
                      oldPeriphery: Array[LearningNode],
                      bestSplitsAndGains: Array[(Option[YggSplit], ImpurityStats)],
                      minInfoGain: Double): Array[LearningNode] = {
    bestSplitsAndGains.zipWithIndex.flatMap { case ((split, stats), nodeIdx) =>
      val node = oldPeriphery(nodeIdx)
      if (split.nonEmpty && stats.gain > minInfoGain) {
        // TODO: remove node id
        node.leftChild = Some(LearningNode(node.id * 2, isLeaf = false,
          new ImpurityStats(Double.NaN, stats.leftImpurity, stats.leftImpurityCalculator,
            null, null, true)))
        node.rightChild = Some(LearningNode(node.id * 2 + 1, isLeaf = false,
          new ImpurityStats(Double.NaN, stats.rightImpurity, stats.rightImpurityCalculator,
            null, null, true)))
        node.split = Some(split.get.toML)
        node.isLeaf = false
        node.stats = stats
        Iterator(node.leftChild.get, node.rightChild.get)
      } else {
        node.isLeaf = true
        Iterator()
      }
    }
  }

  /**
    * Aggregate bit vector (1 bit/instance) indicating whether each instance goes left/right.
    * - Send chosen splits to workers.
    * - Each worker creates part of the bit vector corresponding to the splits it created.
    * - Aggregate the partial bit vectors to create one vector (of length numRows).
    *   Correction: Aggregate only the pieces of that vector corresponding to instances at
    *   active nodes.
    * @param partitionInfos  RDD with feature data, plus current status metadata
    * @param bestSplits  YggSplit for each active node, or None if that node will not be split
    * @return Array of bit vectors, ordered by offset ranges
    */
  private[impl] def aggregateBitVector(
                    partitionInfos: RDD[PartitionInfo],
                    bestSplits: Array[Option[YggSplit]],
                    numRows: Int): RoaringBitmap = {
    val bestSplitsBc: Broadcast[Array[Option[YggSplit]]] =
      partitionInfos.sparkContext.broadcast(bestSplits)
    val workerBitSubvectors: RDD[RoaringBitmap] = partitionInfos.map {
      case PartitionInfo(columns: Array[FeatureVector], nodeOffsets: Array[Int],
      activeNodes: BitSet, fullImpurities: Array[ImpurityAggregatorSingle]) =>
        val localBestSplits: Array[Option[YggSplit]] = bestSplitsBc.value
        // localFeatureIndex[feature index] = index into PartitionInfo.columns
        val localFeatureIndex: Map[Int, Int] = columns.map(_.featureIndex).zipWithIndex.toMap
        val bitSetForNodes: Iterator[RoaringBitmap] = activeNodes.iterator
          .zip(localBestSplits.iterator).flatMap {
          case (nodeIndexInLevel: Int, Some(split: YggSplit)) =>
            if (localFeatureIndex.contains(split.featureIndex)) {
              // This partition has the column (feature) used for this split.
              val fromOffset = nodeOffsets(nodeIndexInLevel)
              val toOffset = nodeOffsets(nodeIndexInLevel + 1)
              val colIndex: Int = localFeatureIndex(split.featureIndex)
              Iterator(bitVectorFromSplit(columns(colIndex), fromOffset, toOffset, split, numRows))
            } else {
              Iterator()
            }
          case (nodeIndexInLevel: Int, None) =>
            // Do not create a bitVector when there is no split.
            // PartitionInfo.update will detect that there is no
            // split, by how many instances go left/right.
            Iterator()
        }
        if (bitSetForNodes.isEmpty) {
          new RoaringBitmap()
        } else {
          bitSetForNodes.reduce[RoaringBitmap] { (acc, bitv) => acc.or(bitv); acc }
        }
    }
    val aggBitVector: RoaringBitmap = workerBitSubvectors.reduce { (acc, bitv) =>
      acc.or(bitv)
      acc
    }
    bestSplitsBc.unpersist()
    aggBitVector
  }

  /**
    * For a given feature, for a given node, apply a split and return a bit vector indicating the
    * outcome of the split for each instance at that node.
    *
    * @param col  Column for feature
    * @param from  Start offset in col for the node
    * @param to  End offset in col for the node
    * @param split  YggSplit to apply to instances at this node.
    * @return  Bits indicating splits for instances at this node.
    *          These bits are sorted by the row indices, in order to guarantee an ordering
    *          understood by all workers.
    *          Thus, the bit indices used are based on 2-level sorting: first by node, and
    *          second by sorted row indices within the node's rows.
    *          bit[index in sorted array of row indices] = false for left, true for right
    */
  private[impl] def bitVectorFromSplit(
                    col: FeatureVector,
                    from: Int,
                    to: Int,
                    split: YggSplit,
                    numRows: Int): RoaringBitmap = {
    val bitv = new RoaringBitmap()
    var i = from
    while (i < to) {
      val value = col.values(i)
      val idx = col.indices(i)
      if (!split.shouldGoLeft(value)) {
        bitv.add(idx)
      }
      i += 1
    }
    bitv
  }

}
