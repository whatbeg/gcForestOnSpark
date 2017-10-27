/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package examples.Sequence

import org.apache.spark.ml.classification.GCForestClassifier
import datasets.UCI_adult


object GCForestSequence {
  def main(args: Array[String]): Unit = {

    val output = "data/uci_adult/model"

    val train = new UCI_adult().load_data("train", 1)

    val test = new UCI_adult().load_data("test", 1)

    train.sparkSession.sparkContext.setLogLevel("ERROR")

    val gcForest = new GCForestClassifier()
      .setDataSize(Array(113))
      .setDataStyle("sequence")
      .setMultiScanWindow(Array())
      .setCascadeForestTreeNum(500)
      .setScanForestTreeNum(1)
      .setMaxIteration(1)
      .setEarlyStoppingRounds(4)

    val model = gcForest.fit(train, test)
    // model.save(output)
  }
}

