/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.classification

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.YggdrasilImpl
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable, Instrumentation, MetadataUtils}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

class YggdrasilClassifier (override val uid: String)
  extends ProbabilisticClassifier[Vector, YggdrasilClassifier, DecisionTreeClassificationModel]
  with DecisionTreeClassifierParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("yggd"))

  // Override parameter setters from parent trait for Java API compatibility.
  override def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  override def setMaxBins(value: Int): this.type = set(maxBins, value)

  override def setMinInstancesPerNode(value: Int): this.type = set(minInstancesPerNode, value)

  override def setMinInfoGain(value: Double): this.type = set(minInfoGain, value)

  override def setMaxMemoryInMB(value: Int): this.type = set(maxMemoryInMB, value)

  override def setCacheNodeIds(value: Boolean): this.type = set(cacheNodeIds, value)

  override def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  override def setImpurity(value: String): this.type = set(impurity, value)

  override def setSeed(value: Long): this.type = set(seed, value)

  override def copy(extra: ParamMap): YggdrasilClassifier = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): DecisionTreeClassificationModel = {
    val categoricalFeatures: Map[Int, Int] =
    MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val numClasses: Int = getNumClasses(dataset)

    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".train() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }

    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset, numClasses)
    val strategy =
      super.getOldStrategy(categoricalFeatures, numClasses, OldAlgo.Classification, getOldImpurity, 1.0)

    val instr = Instrumentation.create(this, oldDataset)
    instr.logParams(params: _*)

    val model = train(oldDataset, strategy, colStoreInput = None, parentUID = Some(uid))
    model.asInstanceOf[DecisionTreeClassificationModel]
  }

  /**
    * Method to train a decision tree model over an RDD.
    */
  def train(
       input: RDD[LabeledPoint],
       strategy: Strategy,
       colStoreInput: Option[RDD[(Int, Array[Double])]] = None,
       parentUID: Option[String] = None): DecisionTreeModel = {
    // TODO: Check validity of params
    // TODO: Check for empty dataset
    YggdrasilImpl.trainImpl(input, strategy, colStoreInput, parentUID)
  }
}
















