/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package examples.UCI_adult

import org.apache.spark.ml.classification.{GCForestClassifier, RandomForestCARTClassifier}
import datasets.UCI_adult
import org.apache.spark.sql.SparkSession
import org.apache.spark.util.SizeEstimator
import org.apache.spark.utils.engine.Engine


object GCForestSequence {
  def main(args: Array[String]): Unit = {

    import Utils._
    val stime = System.currentTimeMillis()
    val spark = SparkSession.builder()
      .appName(this.getClass.getSimpleName)
//      .master("local[*]")
      .getOrCreate()

    val parallelism = Engine.getParallelism(spark.sparkContext)
    println(s"Create Spark Context Succeed! Parallelism is $parallelism")
    spark.conf.set("spark.default.parallelism", parallelism)
    spark.conf.set("spark.locality.wait.node", 0)
//    spark.conf.set("spark.history.fs.logDirectory", "/tmp/spark-events")
    spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    spark.sparkContext.getConf.registerKryoClasses(Array(classOf[RandomForestCARTClassifier]))

    trainParser.parse(args, TrainParams()).map(param => {

      spark.sparkContext.setLogLevel(param.debugLevel)

      val output = param.model

      val train = new UCI_adult().load_data(spark, param.trainFile, param.featuresFile, 1, parallelism)
      val test = new UCI_adult().load_data(spark, param.testFile, param.featuresFile, 1, parallelism)
      if (param.idebug) println(s"Estimate trainset ${SizeEstimator.estimate(train) / (1024 * 1024.0)} M," +
        s" testset: ${SizeEstimator.estimate(test) / (1024 * 1024.0)} M")

      val gcForest = new GCForestClassifier()
        .setDataSize(param.dataSize)
        .setDataStyle(param.dataStyle)
        .setMultiScanWindow(param.multiScanWindow)
        .setRFNum(param.rfNum)
        .setCRFNum(param.crfNum)
        .setCascadeForestTreeNum(param.cascadeForestTreeNum)
        .setScanForestTreeNum(param.scanForestTreeNum)
        .setMaxIteration(param.maxIteration)
        .setMaxDepth(param.maxDepth)
        .setMaxBins(param.maxBins)
        .setMinInfoGain(param.minInfoGain)
        .setMaxMemoryInMB(param.maxMemoryInMB)
        .setCacheNodeId(param.cacheNodeId)
        .setScanForestMinInstancesPerNode(param.scanMinInsPerNode)
        .setCascadeForestMinInstancesPerNode(param.cascadeMinInsPerNode)
        .setEarlyStoppingRounds(param.earlyStoppingRounds)
        .setIDebug(param.idebug)
      if (param.idebug) println(s"Estimate GCForestClassifier ${SizeEstimator.estimate(gcForest) / (1024 * 1024.0)} M")
      val model = gcForest.train(train, test)
      // model.save(output)
      model
    })
    val totalTime = (System.currentTimeMillis() - stime) / 1000.0
    println(s"Total time for GCForest Application: $totalTime")
    spark.stop()
  }
}

