/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.evaluation

import org.apache.spark.ml.Utils.SparkUnitTest
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import Evaluator.evaluate

class EvaluatorSpec extends SparkUnitTest {
  test("evaluate") {
    val dataset = spark.createDataFrame(Seq(
      (0L, 0.0, new DenseVector(Array(0.6, 0.4))),
      (2L, 1.0, new DenseVector(Array(0.3, 0.7))),
      (1L, 1.0, new DenseVector(Array(0.8, 0.2))),
      (3L, 0.0, new DenseVector(Array(0.23, 0.77)))
    )).toDF("instance", "label", "features")
    val acc = evaluate(dataset)
    assert(acc.equals(new Accuracy(2.0, 4.0)))
  }

}
