/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package examples.Sequence

import org.apache.spark.ml.classification.GCForestClassifier
import datasets.UCI_adult
import org.apache.spark.sql.SparkSession


object GCForestSequence {
  def main(args: Array[String]): Unit = {

    val output = "data/uci_adult/model"

    val spark = SparkSession.builder()
      .appName(this.getClass.getSimpleName)
      .master("local[*]").getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    val train = new UCI_adult().load_data(spark, "train", 1)

    val test = new UCI_adult().load_data(spark, "test", 1)

    val gcForest = new GCForestClassifier()
      .setDataSize(Array(113))
      .setDataStyle("sequence")
      .setMultiScanWindow(Array())
      .setCascadeForestTreeNum(500)
      .setScanForestTreeNum(1)
      .setMaxIteration(2)
      .setEarlyStoppingRounds(4)

    val model = gcForest.fit(train, test)
    // model.save(output)
  }
}

