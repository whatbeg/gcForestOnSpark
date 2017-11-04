/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package examples.UCI_adult

import org.apache.spark.ml.classification.GCForestClassifier
import datasets.UCI_adult
import org.apache.spark.sql.SparkSession


object GCForestSequence {
  def main(args: Array[String]): Unit = {

    import Utils._


    val spark = SparkSession.builder()
      .appName(this.getClass.getSimpleName)
      .config("spark.executor.memory", "2g")
      .config("spark.driver.memory", "2g")
      .master("local[8]")
      .getOrCreate()

    println(s"Create Spark Context Succeed! Parallelism: ${spark.sparkContext.defaultParallelism}")

    trainParser.parse(args, TrainParams()).map(param => {

      spark.sparkContext.setLogLevel(param.debugLevel)

      val output = param.model

      val train = new UCI_adult().load_data(spark, param.trainFile, 1)
        .repartition(spark.sparkContext.defaultParallelism)

      val test = new UCI_adult().load_data(spark, param.testFile, 1)
        .repartition(spark.sparkContext.defaultParallelism)

      val gcForest = new GCForestClassifier()
        .setDataSize(param.dataSize)
        .setDataStyle(param.dataStyle)
        .setMultiScanWindow(param.multiScanWindow)
        .setCascadeForestTreeNum(param.cascadeForestTreeNum)
        .setScanForestTreeNum(param.scanForestTreeNum)
        .setMaxIteration(param.maxIteration)
        .setEarlyStoppingRounds(param.earlyStoppingRounds)

      val model = gcForest.train(train, test)
      // model.save(output)
      model
    })
    spark.stop()
  }
}

