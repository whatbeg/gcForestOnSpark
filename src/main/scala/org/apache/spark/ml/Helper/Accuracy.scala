/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package org.apache.spark.ml.Helper

class Accuracy(right: Long, total: Long) {
  private val denom = if (total <= 0) 1.0d else total.toDouble
  private val accuracy = right / denom

  def getAccuracy: Double = accuracy

  override def toString: String = {
    s"Accuracy($right / $total = %.3f%%)".format(accuracy * 100.0)
  }
}
