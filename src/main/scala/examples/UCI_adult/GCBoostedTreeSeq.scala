/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package examples.UCI_adult

import org.apache.spark.ml.classification.GCBoostedTreeClassifier
import datasets.UCI_adult
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.util.engine.Engine


object GCBoostedTreeSeq {
  def main(args: Array[String]): Unit = {

    import GBTUtils._

    val spark = SparkSession.builder()
      .appName(this.getClass.getSimpleName)
      .master("local[*]")
      .getOrCreate()

    val parallelism = Engine.getParallelism(spark.sparkContext)
    println(s"Create Spark Context Succeed! Parallelism is $parallelism")
    spark.conf.set("spark.default.parallelism", parallelism)
    spark.conf.set("spark.locality.wait.node", 0)

    trainParser.parse(args, TrainParams()).map(param => {

      spark.sparkContext.setLogLevel(param.debugLevel)

      val output = param.model

      val train = new UCI_adult().load_data(spark, param.trainFile, param.featuresFile, 1, parallelism)
      val test = new UCI_adult().load_data(spark, param.testFile, param.featuresFile, 1, parallelism)

      val gbtForest = new GCBoostedTreeClassifier()
        .setModelPath(param.model)
        .setDataSize(param.dataSize)
        .setDataStyle(param.dataStyle)
        .setMultiScanWindow(param.multiScanWindow)
        .setCascadeForestTreeNum(param.cascadeForestTreeNum)
        .setScanForestTreeNum(param.scanForestTreeNum)
        .setMaxIteration(param.maxIteration)
        .setMaxDepth(param.maxDepth)
        .setMaxBins(param.maxBins)
        .setMaxMemoryInMB(param.maxMemoryInMB)
        .setCacheNodeId(param.cacheNodeId)
        .setScanForestMinInstancesPerNode(param.scanMinInsPerNode)
        .setCascadeForestMinInstancesPerNode(param.cascadeMinInsPerNode)
        .setEarlyStoppingRounds(param.earlyStoppingRounds)
        .setIDebug(param.idebug)

      val model = gbtForest.train(train, test)
      // model.save(output)
      model
    })
    spark.stop()
  }
}

