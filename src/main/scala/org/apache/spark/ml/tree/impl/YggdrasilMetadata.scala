/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.tree.impl

import org.apache.spark.ml.tree._
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.impurity.{Entropy, Gini, Impurity, Variance}

private[spark] class YggdrasilMetadata(
     val numClasses: Int,
     val maxBins: Int,
     val minInfoGain: Double,
     val impurity: Impurity,
     val categoricalFeaturesInfo: Map[Int, Int]) extends Serializable {

  private val unorderedSplits = {
    /**
      * borrowed from [[DecisionTreeMetadata.buildMetadata]]
      */
    if (numClasses > 2) {
      // Multiclass classification
      val maxCategoriesForUnorderedFeature =
        ((math.log(maxBins / 2 + 1) / math.log(2.0)) + 1).floor.toInt
      categoricalFeaturesInfo.filter { case (featureIndex, numCategories) =>
        numCategories > 1 && numCategories <= maxCategoriesForUnorderedFeature
      }.map { case (featureIndex, numCategories) =>
        // Hack: If a categorical feature has only 1 category, we treat it as continuous.
        // TODO(SPARK-9957): Handle this properly by filtering out those features.
        // Decide if some categorical features should be treated as unordered features,
        //  which require 2 * ((1 << numCategories - 1) - 1) bins.
        // We do this check with log values to prevent overflows in case numCategories is large.
        // The next check is equivalent to: 2 * ((1 << numCategories - 1) - 1) <= maxBins
        featureIndex -> findSplits(featureIndex, numCategories)
      }
    } else {
      Map.empty[Int, Array[CategoricalSplit]]
    }
  }

  /**
    * Returns all possible subsets of features for categorical splits.
    * Borrowed from [[RandomForest.findSplits]]
    */
  private def findSplits(
                          featureIndex: Int,
                          featureArity: Int): Array[CategoricalSplit] = {
    // Unordered features
    // 2^(featureArity - 1) - 1 combinations
    val numSplits = (1 << (featureArity - 1)) - 1
    val splits = new Array[CategoricalSplit](numSplits)

    var splitIndex = 0
    while (splitIndex < numSplits) {
      val categories: List[Double] =
        RandomForest.extractMultiClassCategories(splitIndex + 1, featureArity)
      splits(splitIndex) =
        new CategoricalSplit(featureIndex, categories.toArray, featureArity)
      splitIndex += 1
    }
    splits
  }

  def getUnorderedSplits(featureIndex: Int): Array[CategoricalSplit] = unorderedSplits(featureIndex)

  def isClassification: Boolean = numClasses >= 2

  def isMulticlass: Boolean = numClasses > 2

  def isUnorderedFeature(featureIndex: Int): Boolean = unorderedSplits.contains(featureIndex)

  def createImpurityAggregator(): ImpurityAggregatorSingle = {
    impurity match {
      case Entropy => new EntropyAggregatorSingle(numClasses)
      case Gini => new GiniAggregatorSingle(numClasses)
      case Variance => new VarianceAggregatorSingle
    }
  }
}

private[spark] object YggdrasilMetadata {
  def fromStrategy(strategy: Strategy): YggdrasilMetadata = new YggdrasilMetadata(strategy.numClasses,
    strategy.maxBins, strategy.minInfoGain, strategy.impurity, strategy.categoricalFeaturesInfo)
}
