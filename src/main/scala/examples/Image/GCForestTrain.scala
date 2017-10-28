/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package examples.Image

import org.apache.spark.ml.classification.GCForestClassifier
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, DoubleType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}


object GCForestTrain {
  def main(args: Array[String]): Unit = {

    val input = "data/sample_data.txt"
    val output = "data/model"

    val spark = SparkSession.builder()
      .appName(this.getClass.getSimpleName)
      .master("local[*]").getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    val raw = spark.read.text(input)

    val trainRDD = raw.rdd.map { row =>
      val data = row.getString(0).split(",")
      val label = if (data(1) == "cat") 0.0 else 1.0
      val features = data.drop(2).map(_.toDouble)

      Row.fromSeq(Seq[Any](label, features))
    }

    val schema = new StructType()
      .add(StructField("label", DoubleType))
      .add(StructField("features", ArrayType(DoubleType)))

    val arr2vec = udf {(features: Seq[Double]) => new DenseVector(features.toArray)}
    val train = spark.createDataFrame(trainRDD, schema)
      .withColumn("features", arr2vec(col("features")))

    val gcForest = new GCForestClassifier()
      .setDataSize(Array(10, 10))
      .setDataStyle("image")
      .setMultiScanWindow(Array(9, 9))
      .setCascadeForestTreeNum(1)
      .setScanForestTreeNum(1)
      .setMaxIteration(1)

    // val commonForest = new CompleteRandomTreeForestClassifier()
    val model = gcForest.fit(train)
    // val model = commonForest.fit(train)
    // model.save(output)
  }
}
