/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.tree

import org.apache.spark.ml.Utils.SparkUnitTest
import org.apache.spark.ml.classification.GCForestClassifier
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.tree.impl.GCForestImpl

class GCForestImplSuite extends SparkUnitTest {
  test("mergeFeatureAndPredict") {
    val features = spark.createDataFrame(Seq(
      (0L, 0.0, new DenseVector(Array(1, 4))),
      (2L, 1.0, new DenseVector(Array(3, 7))),
      (1L, 1.0, new DenseVector(Array(8, 2))),
      (3L, 0.0, new DenseVector(Array(23, 77)))
    )).toDF("instance", "label", "features")

    val predict = spark.createDataFrame(Seq(
      (0L, new DenseVector(Array(0.6, 0.4))),
      (2L, new DenseVector(Array(0.3, 0.7))),
      (1L, new DenseVector(Array(0.8, 0.2))),
      (3L, new DenseVector(Array(0.23, 0.77)))
    )).toDF("instance", "features")

    val strategy = new GCForestClassifier().getDefaultStrategy

    val merged = GCForestImpl.mergeFeatureAndPredict(features, predict, strategy)

    val shouldMerged = spark.createDataFrame(Seq(
      (0L, new DenseVector(Array(1.0, 4.0, 0.6, 0.4)), 0.0),
      (2L, new DenseVector(Array(3.0, 7.0, 0.3, 0.7)), 1.0),
      (1L, new DenseVector(Array(8.0, 2.0, 0.8, 0.2)), 1.0),
      (3L, new DenseVector(Array(23, 77, 0.23, 0.77)), 0.0)
    )).toDF("instance", "features", "label")

    assert(merged.schema == shouldMerged.schema)
    merged.rdd.zip(shouldMerged.rdd).foreach { case (row1, row2) =>
      require(row1.mkString(",") == row2.mkString(","))
    }
  }
}
