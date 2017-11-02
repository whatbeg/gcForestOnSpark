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
      .config("spark.executor.memory", "2g")
      .config("spark.driver.memory", "2g")
//      .config("spark.sql.shuffle.partitions", "8")
//      .config("spark.default.parallelism", "8")
//      .config("spark.storage.memoryFraction", "0.4")
//      .master("local-cluster[2,4,2048]")
      .master("local[8]")
      .getOrCreate()

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

