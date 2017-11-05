/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package examples.UCI_adult

import org.apache.spark.ml.classification.GCForestClassifier
import datasets.UCI_adult
import org.apache.spark.sql.SparkSession
import org.apache.spark.utils.engine.Engine


object GCForestSequence {
  def main(args: Array[String]): Unit = {

    import Utils._


    val spark = SparkSession.builder()
      .appName(this.getClass.getSimpleName)
//      .master("local[*]")
      .getOrCreate()

//    println(spark.conf.getAll)
//    println(s"Engine getParallelism: ${Engine.getParallelism(spark.sparkContext)}")
//    println(s"Create Spark Context Succeed! Parallelism: ${spark.sparkContext.defaultParallelism}")
    val parallelism = Engine.getParallelism(spark.sparkContext)
    println(s"Create Spark Context Succeed! Parallelism is $parallelism")
    spark.conf.set("spark.default.parallelism", parallelism)
    spark.conf.set("spark.locality.wait.node", 0)

    trainParser.parse(args, TrainParams()).map(param => {

      spark.sparkContext.setLogLevel(param.debugLevel)

      val output = param.model

      val train = new UCI_adult().load_data(spark, param.trainFile, param.featuresFile, 1)
        .repartition(parallelism)
      // if (param.idebug) println(s"train repartition ${spark.sparkContext.defaultParallelism}")
      val test = new UCI_adult().load_data(spark, param.testFile, param.featuresFile, 1)
        .repartition(parallelism)
      // if (param.idebug) println(s"test repartition ${spark.sparkContext.defaultParallelism}")
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

