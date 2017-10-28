/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package org.apache.spark.ml.evaluation

class Accuracy(right: Double, total: Double) extends Metric {
  val rightCount: Double = right
  val totalCount: Double = total
  private val denom = if (total <= 0) 1.0 else total
  private val accuracy = right / denom

  def getAccuracy: Double = accuracy

  def +(that: Accuracy): Accuracy = {
    val new_right = this.rightCount + that.rightCount
    val new_total = this.totalCount + that.totalCount
    new Accuracy(new_right, new_total)
  }

  override def +(that: Metric): Accuracy = {
    require(that.isInstanceOf[Accuracy], "Not an Accuracy Object")
    new Accuracy(this.rightCount + that.asInstanceOf[Accuracy].rightCount,
      this.totalCount + that.asInstanceOf[Accuracy].totalCount)
  }

  override def /(value: Double): Accuracy = {
    new Accuracy(this.rightCount / value, this.totalCount)
  }

  override def toString: String = {
    s"Accuracy($rightCount / $totalCount = %.3f%%)".format(accuracy * 100.0)
  }

  def equals(obj: Accuracy): Boolean = {
    (this.rightCount == obj.rightCount) && (this.totalCount == obj.totalCount)
  }
}
