/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package org.apache.spark.ml.evaluation

class Accuracy(private var rightCount: Double, private var totalCount: Double) extends Metric {
  private val denom = if (totalCount <= 0) 1.0 else totalCount
  private val accuracy = rightCount / denom

  def getAccuracy: Double = accuracy

  def +(that: Accuracy): Accuracy = {
    this.rightCount += that.rightCount
    this.totalCount += that.totalCount
    this
  }

  override def +(that: Metric): Accuracy = {
    require(that.isInstanceOf[Accuracy], "Not an Accuracy Object")
    this.rightCount += that.asInstanceOf[Accuracy].rightCount
    this.totalCount += that.asInstanceOf[Accuracy].totalCount
    this
  }

  override def /(value: Double): Accuracy = {
    this.rightCount /= value
    this
  }

  def div(value: Double): Accuracy = {
    this.rightCount /= value
    this.totalCount /= value
    this
  }

  override def toString: String = {
    s"Accuracy($rightCount / $totalCount = %.3f%%)".format(accuracy * 100.0)
  }

  def equals(obj: Accuracy): Boolean = {
    (this.rightCount == obj.rightCount) && (this.totalCount == obj.totalCount)
  }

  def reset(): Accuracy = {
    this.rightCount = 0.0
    this.totalCount = 0.0
    this
  }
}
