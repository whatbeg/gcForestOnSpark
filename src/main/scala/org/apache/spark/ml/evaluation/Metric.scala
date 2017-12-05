/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.evaluation

import org.apache.spark.internal.Logging

abstract class Metric extends Logging {
  def +(that: Metric): Metric
  def /(value: Double): Metric

}
