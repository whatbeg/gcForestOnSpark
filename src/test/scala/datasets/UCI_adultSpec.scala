/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package datasets

import org.apache.spark.ml.Utils.SparkUnitTest

class UCI_adultSpec extends SparkUnitTest {

  val data = new UCI_adult().load_data(spark, "train", "", 1)
  val list = data.collectAsList()
  println(list)
}
