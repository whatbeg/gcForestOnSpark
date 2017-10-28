/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.evaluation

import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.types.{LongType, StructField}


abstract class Evaluator {
  /**
    * Evaluates model output and returns a metric.
    *
    * @param dataset dataset to evaluate
    * @return metric
    */
  def evaluate(dataset: Dataset[_]): Metric

}


object Evaluator {

  case class Record(instance: Long, label: Double, features: Vector)
  /**
    * Evaluates model output and returns a Accuracy metric.
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

    val rightCount = dataset.rdd.map { case (row: Row) =>
      val features = row.getAs[Vector]("features")
      val label = row.getAs[Double]("label")
      if (features.argmax == label.toInt) 1d else 0d
    }.filter(_ > 0).count()
    val totalCount = dataset.count()
    new Accuracy(rightCount, totalCount)
  }
}