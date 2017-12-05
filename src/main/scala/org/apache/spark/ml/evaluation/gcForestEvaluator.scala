/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.evaluation

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Dataset, Row}


abstract class gcForestEvaluator {
  /**
    * Evaluates model output and returns a metric.
    *
    * @param dataset dataset to evaluate
    * @return metric
    */
  def evaluate(dataset: Dataset[_]): Metric

}


object gcForestEvaluator {

  /**
    * Evaluates model output (instance: Long, label: Double, features: Vector)
    * and returns a Accuracy metric.
    *
    * @param dataset dataset to evaluate
    * @return metric
    */
  def evaluate(dataset: Dataset[_]): Accuracy = {
    val schema = dataset.schema
    val schema_string = schema.toArray.map { sch => sch.name }
    require(schema_string.contains("instance"), "DataSet has no instance col")
    require(schema_string.contains("label"), "DataSet has no label col")
    require(schema_string.contains("features"), "DataSet has no features col")

    val totalCount = dataset.sparkSession.sparkContext.doubleAccumulator("totalCount")
    val rightCount = dataset.sparkSession.sparkContext.doubleAccumulator("rightCount")
    dataset.rdd.foreach { case (row: Row) =>
      val features = row.getAs[Vector]("features")
      val label = row.getAs[Double]("label")
      if (features.argmax == label.toInt) rightCount.add(1)
      totalCount.add(1)
    }
    new Accuracy(rightCount.value, totalCount.value)
  }

  // DONE at 11.28 21:05
  def evaluatePartition(dataset: Dataset[_]): Accuracy = {
    val schema = dataset.schema
    val schema_string = schema.toArray.map { sch => sch.name }
    require(schema_string.contains("instance"), "DataSet has no instance col")
    require(schema_string.contains("label"), "DataSet has no label col")
    require(schema_string.contains("features"), "DataSet has no features col")

    val eval_array = dataset.rdd.mapPartitions { iter =>
      var agg = 0
      var total = 0
      iter.foreach { case (row: Row) =>
        val features = row.getAs[Vector]("features")
        val label = row.getAs[Double]("label")
        if (features.argmax == label.toInt) agg += 1
        total += 1
      }
      Iterator.single((agg, total))
    }.collect()
    val tup = eval_array.foldLeft((0, 0)) { case (t, x) =>
      (t._1 + x._1, t._2 + x._2)
    }
    new Accuracy(tup._1, tup._2)
  }

  def evaluatePrediction(dataset: Dataset[_]): Accuracy = {
    val schema = dataset.schema
    val schema_string = schema.toArray.map { sch => sch.name }
    require(schema_string.contains("label"), "DataSet has no label col")
    require(schema_string.contains("prediction"), "DataSet has no prediction col")
    val totalCount = dataset.sparkSession.sparkContext.doubleAccumulator("totalCount")
    val rightCount = dataset.sparkSession.sparkContext.doubleAccumulator("rightCount")
    dataset.rdd.foreach { case (row: Row) =>
      val predict = row.getAs[Double]("prediction")
      val label = row.getAs[Double]("label")
      if (predict.toInt == label.toInt) rightCount.add(1)
      totalCount.add(1)
    }
    new Accuracy(rightCount.value, totalCount.value)
  }
}
