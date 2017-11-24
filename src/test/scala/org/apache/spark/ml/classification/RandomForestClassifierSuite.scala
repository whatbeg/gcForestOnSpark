/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.classification

import java.io.File

import datasets.UCI_adult
import org.apache.spark.ml.Utils.SparkUnitTest
import org.apache.spark.ml.evaluation.{Evaluator, gcForestEvaluator}

import scala.util.Random


class RandomForestClassifierSuite extends SparkUnitTest {
  test("Random Forest is really Random") {
    val resource = getClass.getClassLoader.getResource("test-data")
    val dataPrefix = resource.getPath + File.separator
    val training = new UCI_adult().load_data(spark, dataPrefix + "sample_adult.data", dataPrefix + "features", 1)
    val testing = new UCI_adult().load_data(spark, dataPrefix + "sample_adult.test", dataPrefix + "features", 1)
    val acc_list = Range(0, 4).map { r =>
      val rf = new RandomForestClassifier()
        .setNumTrees(1)
        .setMaxBins(32)
        .setMaxDepth(30)
        .setMinInstancesPerNode(1)
        .setFeatureSubsetStrategy("sqrt")
        .setSeed(Random.nextInt() + r)
      val model = rf.fit(training)
      val test_result = model.transform(testing)
        .drop("features").drop("rawPrediction").drop("prediction")
        .withColumnRenamed("probability", "features")
      val acc = gcForestEvaluator.evaluate(test_result)
      println(acc)
      acc.getAccuracy
    }
    assert(acc_list.distinct.length == acc_list.length)
  }
}
