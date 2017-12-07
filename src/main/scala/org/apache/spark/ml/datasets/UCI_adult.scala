/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package org.apache.spark.ml.datasets

import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._

import scala.io.Source


class UCI_adult extends BaseDatasets {
  /**
    * Load UCI ADULT data, by sparkSession, phase(or file path) and cate_as_onehot
    *
    * @param spark SparkSession to load
    * @param phase which kind of data to load, "train" or "test", or provide file path directly
    * @param cate_as_onehot convert categorical data to one-hot format
    * @return loaded DataFrame
    */
  def load_data(spark: SparkSession,
                phase: String,
                featuresPath: String,
                cate_as_onehot: Boolean,
                repar: Int = 0): DataFrame = {
    val data_path =
      if (phase == "train") "data/uci_adult/sample_adult.data"
      else if (phase == "test") "data/uci_adult/sample_adult.test"
      else phase

    val raw = spark.read.text(data_path)

    val features_path = if (featuresPath == "") "data/uci_adult/features" else featuresPath

    val fts_file = spark.read.text(features_path)
    val f_parsers = fts_file.rdd.filter(row => row.length > 0).map { row =>
      val line = row.getAs[String]("value")
      new FeatureParser(line)
    }

    val total_dims = if (cate_as_onehot) f_parsers.map { fts =>
      fts.get_fdim()
    }.reduce((l, r) => l+r) else 14

    val f_parsers_array = f_parsers.collect()

    val dataRDD = raw.rdd.filter(row => row.mkString.length > 1 && !row.mkString.startsWith("|"))
      .zipWithIndex.map { case (row, idx) =>
      val line = row.getAs[String]("value")
      val splits = line.split(",")
      require(splits.length == 15, s"row $idx: $line has no 15 features, length: ${row.length}")
      val label = if (splits(14).contains("<=50K")) 0.0d else 1.0d
      val data = if (cate_as_onehot) splits.dropRight(1).zipWithIndex.map { case (feature, indx) =>
        f_parsers_array(indx).get_data(feature.trim)
      }.reduce((l, r) => l ++ r)
      else splits.dropRight(1).zipWithIndex.map { case (feature, indx) =>
        f_parsers_array(indx).get_double(feature.trim)
      }
      require(data.length == total_dims,
        "Total dims %d not equal to data.length %d".format(total_dims, data.length))
      Row.fromSeq(Seq[Any](label, data, idx))
    }

    val repartitioned = if (repar > 0) dataRDD.repartition(repar) else dataRDD
    val schema = new StructType()
      .add(StructField("label", DoubleType))
      .add(StructField("features", ArrayType(DoubleType)))
      .add(StructField("instance", LongType))

    val arr2vec = udf { (features: Seq[Double]) => new DenseVector(features.toArray) }
    val return_data = spark.createDataFrame(repartitioned, schema)
      .withColumn("features", arr2vec(col("features")))
    if (!cate_as_onehot)
      return_data.schema.fields.update(1, StructField("features", new VectorUDT, true,
        Metadata.fromJson(Source.fromFile("data/features.json").mkString)))
    return_data
  }
}

class FeatureParser(row: String) extends Serializable {
  private val desc = row.trim
  private val f_type = if (desc == "C") "number" else "categorical"
  private val name_to_len = if (f_type == "categorical") {
    val f_names = Array("?") ++ desc.trim.split(",").map(str => str.trim)
    f_names.zipWithIndex.map { case(cate_name, idx) =>
      cate_name -> idx
    }.toMap
  } else Map[String, Int]()

  def get_double(f_data: String): Double = {
    if (f_type == "number") f_data.trim.toDouble
    else name_to_len.getOrElse(f_data.trim, 0).toDouble
  }

  def get_data(f_data: String): Array[Double] = {
    if (f_type == "number") Array[Double](f_data.trim.toDouble)
    else {
      val data = Array.fill[Double](name_to_len.size)(0f)
      data(name_to_len.getOrElse(f_data.trim, 0)) = 1f
      data
    }
  }

  def get_fdim(): Int = {
    if (f_type == "number") 1 else name_to_len.size
  }
}