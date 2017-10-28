/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.evaluation

abstract class Metric {
  def +(that: Metric): Metric
  def /(value: Double): Metric
}
