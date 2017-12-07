/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package org.apache.spark.ml.datasets

import org.apache.spark.ml.Utils.SparkUnitTest
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.linalg.VectorUDT
import org.apache.spark.sql.catalyst.expressions.StructsToJson
import org.apache.spark.sql.types._

import scala.io.Source

class UCI_adultSpec extends SparkUnitTest {

  test("UCI_adult") {
    val data = new UCI_adult().load_data(spark, "train", "", false)
    data.printSchema()
    data.schema.fields.foreach { k =>
      println(k.metadata)
    }
    // val list = data.collectAsList()
    // println(list)
  }
  test("vectorIndexer does right") {
    val new_data = Seq(
      Vectors.dense(0.0, 1.0, -2.0, 3.0),
      Vectors.dense(-1.0, 2.0, 4.0, 3.0),
      Vectors.dense(14.0, 2.0, -5.0, 1.0))

    val tup = new_data.map(Tuple1.apply)
//    println(tup)

    val new_dataframe = spark.createDataFrame(tup).toDF("features")
//    new_dataframe.show()
//    new_dataframe.printSchema()
//    new_dataframe.schema.fields.foreach { k =>
//      println(k.metadata)
//    }
    val vectorIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("featuresIdx")
      .setMaxCategories(2)
      .fit(new_dataframe)
      .transform(new_dataframe)

    vectorIndexer.show()
    vectorIndexer.printSchema()
    vectorIndexer.schema.fields.foreach { k =>
      println(k.metadata)
    }
  }

  test("VectorAssembler") {
    val new_data = Seq(
      (0.0, 1.0, -2.0, 3.0),
      (-1.0, 2.0, 4.0, 3.0),
      (14.0, 2.0, -5.0, 1.0)
    )

    val tup = new_data.map(Tuple1.apply)
    //    println(tup)

    val new_dataframe = spark.createDataFrame(tup).toDF("features")

        new_dataframe.show()
        new_dataframe.printSchema()
    //    new_dataframe.schema.fields.foreach { k =>
    //      println(k.metadata)
    //    }
//    val vectorIndexer = new VectorIndexer()
//      .setInputCol("features")
//      .setOutputCol("featuresIdx")
//      .setMaxCategories(2)
//      .fit(new_dataframe)
//      .transform(new_dataframe)
//
//    vectorIndexer.show()
//    vectorIndexer.printSchema()
//    vectorIndexer.schema.fields.foreach { k =>
//      println(k.metadata)
//    }
  }
}
