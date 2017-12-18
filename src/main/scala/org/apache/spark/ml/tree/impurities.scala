/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.tree

import org.apache.spark.mllib.tree.impurity.{EntropyCalculator, GiniCalculator, ImpurityCalculator,
VarianceCalculator}

/**
  * Version of impurity aggregator which owns its data and is only for 1 node.
  */
private[ml] abstract class ImpurityAggregatorSingle(val stats: Array[Double])
  extends Serializable {

  def statsSize: Int = stats.length

  /**
    * Add two aggregators: this + other
    * @return This aggregator (modified).
    */
  def add(other: ImpurityAggregatorSingle): this.type = {
    var i = 0
    while (i < statsSize) {
      stats(i) += other.stats(i)
      i += 1
    }
    this
  }

  /**
    * Subtract another aggregators from this one: this - other
    * @return This aggregator (modified).
    */
  def subtract(other: ImpurityAggregatorSingle): this.type = {
    var i = 0
    while (i < statsSize) {
      stats(i) -= other.stats(i)
      i += 1
    }
    this
  }

  /**
    * Update stats with the given label and instance weight.
    * @return This aggregator (modified).
    */
  def update(label: Double, instanceWeight: Double): this.type

  /**
    * Update stats with the given label.
    * @return This aggregator (modified).
    */
  def update(label: Double): this.type = update(label, 1.0)

  /** Get an [[ImpurityCalculator]] for the current stats. */
  def getCalculator: ImpurityCalculator

  def deepCopy(): ImpurityAggregatorSingle

  /** Total (weighted) count of instances in this aggregator */
  def getCount: Double

  /** Resets this aggregator as though nothing has been added to it. */
  def clear(): this.type = {
    var i = 0
    while (i < statsSize) {
      stats(i) = 0.0
      i += 1
    }
    this
  }
}

/**
  * Version of Entropy aggregator which owns its data and is only for one node.
  */
private[ml] class EntropyAggregatorSingle private (stats: Array[Double])
  extends ImpurityAggregatorSingle(stats) with Serializable {

  def this(numClasses: Int) = this(new Array[Double](numClasses))

  def update(label: Double, instanceWeight: Double): this.type = {
    if (label >= statsSize) {
      throw new IllegalArgumentException(s"EntropyAggregatorSingle given label $label" +
        s" but requires label < numClasses (= $statsSize).")
    }
    stats(label.toInt) += instanceWeight
    this
  }

  def getCalculator: EntropyCalculator = new EntropyCalculator(stats)

  override def deepCopy(): ImpurityAggregatorSingle = new EntropyAggregatorSingle(stats.clone())

  override def getCount: Double = stats.sum
}

/**
  * Version of Gini aggregator which owns its data and is only for one node.
  */
private[ml] class GiniAggregatorSingle private (stats: Array[Double])
  extends ImpurityAggregatorSingle(stats) with Serializable {

  def this(numClasses: Int) = this(new Array[Double](numClasses))

  def update(label: Double, instanceWeight: Double): this.type = {
    if (label >= statsSize) {
      throw new IllegalArgumentException(s"GiniAggregatorSingle given label $label" +
        s" but requires label < numClasses (= $statsSize).")
    }
    stats(label.toInt) += instanceWeight
    this
  }

  def getCalculator: GiniCalculator = new GiniCalculator(stats)

  override def deepCopy(): ImpurityAggregatorSingle = new GiniAggregatorSingle(stats.clone())

  override def getCount: Double = stats.sum
}

/**
  * Version of Variance aggregator which owns its data and is only for one node.
  */
private[ml] class VarianceAggregatorSingle
  extends ImpurityAggregatorSingle(new Array[Double](3)) with Serializable {

  def update(label: Double, instanceWeight: Double): this.type = {
    stats(0) += instanceWeight
    stats(1) += instanceWeight * label
    stats(2) += instanceWeight * label * label
    this
  }

  def getCalculator: VarianceCalculator = new VarianceCalculator(stats)

  override def deepCopy(): ImpurityAggregatorSingle = {
    val tmp = new VarianceAggregatorSingle()
    stats.copyToArray(tmp.stats)
    tmp
  }

  override def getCount: Double = stats(0)
}
