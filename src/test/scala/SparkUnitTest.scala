/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, FunSuite}


abstract class SparkUnitTest extends FunSuite with BeforeAndAfterAll {
  var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .appName(this.getClass.getSimpleName)
      .master("local[4]")
      .getOrCreate()
  }

  override def afterAll(): Unit = spark.stop()
}
