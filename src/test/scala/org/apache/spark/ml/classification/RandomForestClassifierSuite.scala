/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.classification

import org.apache.spark.ml.Utils.SparkUnitTest
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.linalg.DenseVector

class RandomForestClassifierSuite extends SparkUnitTest {
  test("Random Forest is really Random") {
    val dataset = spark.createDataFrame(Seq(
      (0L, 0.0, new DenseVector(Array(0.6, 0.4))),
      (2L, 1.0, new DenseVector(Array(0.3, 0.7))),
      (1L, 1.0, new DenseVector(Array(0.8, 0.2))),
      (3L, 0.0, new DenseVector(Array(0.23, 0.77)))
    )).toDF("instance", "label", "features")
    val testingDataset = spark.createDataFrame(Seq(
      (0L, 1.0, new DenseVector(Array(0.2, 0.8))),
      (1L, 0.0, new DenseVector(Array(0.7, 0.3)))
    )).toDF("instance", "label", "features")
    val rf = new RandomForestClassifier()
      .setNumTrees(10)
      .setMaxBins(32)
      .setMaxDepth(10)
      .setMinInstancesPerNode(1)
      .setFeatureSubsetStrategy("sqrt")
    val model = rf.fit(dataset)
    val test_result = model.transform(testingDataset)
      .drop("features").drop("rawPrediction").drop("prediction")
      .withColumnRenamed("probability", "features")
    val acc = Evaluator.evaluate(test_result)
    println(acc)
  }
}
