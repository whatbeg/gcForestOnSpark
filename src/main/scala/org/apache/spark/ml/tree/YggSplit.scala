/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.tree

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tree.{Split => MLSplit}
import org.apache.spark.mllib.tree.configuration.{FeatureType => OldFeatureType}
import org.apache.spark.mllib.tree.model.{Split => OldSplit}


/**
  * :: DeveloperApi ::
  * Interface for a "Split," which specifies a test made at a decision tree node
  * to choose the left or right path.
  */
@DeveloperApi
sealed trait YggSplit extends Serializable {

  /** Index of feature which this split tests */
  def featureIndex: Int

  /**
    * Return true (split to left) or false (split to right).
    * @param features  Vector of features (original values, not binned).
    */
  private[ml] def shouldGoLeft(features: Vector): Boolean

  /**
    * Return true (split to left) or false (split to right).
    * @param binnedFeature Binned feature value.
    * @param splits All splits for the given feature.
    */
  private[tree] def shouldGoLeft(binnedFeature: Int, splits: Array[YggSplit]): Boolean

  /**
    * Return true (split to left) or false (split to right).
    * @param feature Feature value (original value, not binned)
    */
  private[tree] def shouldGoLeft(feature: Double): Boolean

  /** Convert to old Split format */
  private[tree] def toOld: OldSplit

  /** Convert to ml Split format */
  private[tree] def toML: MLSplit
}

private[tree] object YggSplit {

  def fromOld(oldSplit: OldSplit, categoricalFeatures: Map[Int, Int]): YggSplit = {
    oldSplit.featureType match {
      case OldFeatureType.Categorical =>
        new YggCategoricalSplit(featureIndex = oldSplit.feature,
          _leftCategories = oldSplit.categories.toArray, categoricalFeatures(oldSplit.feature))
      case OldFeatureType.Continuous =>
        new YggContinuousSplit(featureIndex = oldSplit.feature, threshold = oldSplit.threshold)
    }
  }

  def fromML(mlSplit: MLSplit, categoricalFeatures: Map[Int, Int]): YggSplit = {
    mlSplit match {
      case cate: CategoricalSplit =>
        new YggCategoricalSplit(mlSplit.asInstanceOf[CategoricalSplit].featureIndex,
          mlSplit.asInstanceOf[CategoricalSplit].leftCategories,
          mlSplit.asInstanceOf[CategoricalSplit].numCategories)
      case cont: ContinuousSplit =>
        new YggContinuousSplit(mlSplit.asInstanceOf[ContinuousSplit].featureIndex,
          mlSplit.asInstanceOf[ContinuousSplit].threshold)
    }
  }
}

/**
  * :: DeveloperApi ::
  * Split which tests a categorical feature.
  * @param featureIndex  Index of the feature to test
  * @param _leftCategories  If the feature value is in this set of categories, then the split goes
  *                         left. Otherwise, it goes right.
  * @param numCategories  Number of categories for this feature.
  */
@DeveloperApi
final class YggCategoricalSplit private[ml] (
     override val featureIndex: Int,
     _leftCategories: Array[Double],
     private val numCategories: Int)
  extends YggSplit {

  require(_leftCategories.forall(cat => 0 <= cat && cat < numCategories), "Invalid leftCategories" +
    s" (should be in range [0, $numCategories)): ${_leftCategories.mkString(",")}")

  /**
    * If true, then "categories" is the set of categories for splitting to the left, and vice versa.
    */
  private val isLeft: Boolean = _leftCategories.length <= numCategories / 2

  /** Set of categories determining the splitting rule, along with [[isLeft]]. */
  private val categories: Set[Double] = {
    if (isLeft) {
      _leftCategories.toSet
    } else {
      setComplement(_leftCategories.toSet)
    }
  }

  override private[ml] def shouldGoLeft(features: Vector): Boolean = {
    if (isLeft) {
      categories.contains(features(featureIndex))
    } else {
      !categories.contains(features(featureIndex))
    }
  }

  override private[tree] def shouldGoLeft(binnedFeature: Int, splits: Array[YggSplit]): Boolean = {
    if (isLeft) {
      categories.contains(binnedFeature.toDouble)
    } else {
      !categories.contains(binnedFeature.toDouble)
    }
  }

  override private[tree] def shouldGoLeft(feature: Double): Boolean = {
    if (isLeft) {
      categories.contains(feature)
    } else {
      !categories.contains(feature)
    }
  }

  override def equals(o: Any): Boolean = {
    o match {
      case other: YggCategoricalSplit => featureIndex == other.featureIndex &&
        isLeft == other.isLeft && categories == other.categories
      case _ => false
    }
  }

  override private[tree] def toOld: OldSplit = {
    val oldCats = if (isLeft) {
      categories
    } else {
      setComplement(categories)
    }
    OldSplit(featureIndex, threshold = 0.0, OldFeatureType.Categorical, oldCats.toList)
  }

  override private[tree] def toML: MLSplit = {
    new CategoricalSplit(featureIndex, _leftCategories, numCategories)
  }

  /** Get sorted categories which split to the left */
  def leftCategories: Array[Double] = {
    val cats = if (isLeft) categories else setComplement(categories)
    cats.toArray.sorted
  }

  /** Get sorted categories which split to the right */
  def rightCategories: Array[Double] = {
    val cats = if (isLeft) setComplement(categories) else categories
    cats.toArray.sorted
  }

  /** [0, numCategories) \ cats */
  private def setComplement(cats: Set[Double]): Set[Double] = {
    Range(0, numCategories).map(_.toDouble).filter(cat => !cats.contains(cat)).toSet
  }
}

/**
  * :: DeveloperApi ::
  * Split which tests a continuous feature.
  * @param featureIndex  Index of the feature to test
  * @param threshold  If the feature value is <= this threshold, then the split goes left.
  *                    Otherwise, it goes right.
  */
@DeveloperApi
final class YggContinuousSplit private[ml] (override val featureIndex: Int, val threshold: Double)
  extends YggSplit {

  override private[ml] def shouldGoLeft(features: Vector): Boolean = {
    features(featureIndex) <= threshold
  }

  override private[tree] def shouldGoLeft(binnedFeature: Int, splits: Array[YggSplit]): Boolean = {
    if (binnedFeature == splits.length) {
      // > last split, so split right
      false
    } else {
      val featureValueUpperBound = splits(binnedFeature).asInstanceOf[YggContinuousSplit].threshold
      featureValueUpperBound <= threshold
    }
  }

  override private[tree] def shouldGoLeft(feature: Double): Boolean = {
    feature <= threshold
  }

  override def equals(o: Any): Boolean = {
    o match {
      case other: ContinuousSplit =>
        featureIndex == other.featureIndex && threshold == other.threshold
      case _ =>
        false
    }
  }

  override private[tree] def toOld: OldSplit = {
    OldSplit(featureIndex, threshold, OldFeatureType.Continuous, List.empty[Double])
  }

  override private[tree] def toML: MLSplit = {
    new ContinuousSplit(featureIndex, threshold)
  }
}
