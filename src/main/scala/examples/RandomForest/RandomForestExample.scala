/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package examples.RandomForest

import datasets.UCI_adult
import examples.UCI_adult.Utils
import org.apache.spark.ml.classification.RandomForestCARTClassifier
import org.apache.spark.sql.SparkSession
import org.apache.spark.utils.engine.Engine

object RandomForestExample {
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

      val train = new UCI_adult().load_data(spark, param.trainFile, param.featuresFile, 1, parallelism)
//        .repartition(parallelism)
      // if (param.idebug) println(s"train repartition ${spark.sparkContext.defaultParallelism}")
      val test = new UCI_adult().load_data(spark, param.testFile, param.featuresFile, 1, parallelism)
//        .repartition(parallelism)
      // if (param.idebug) println(s"test repartition ${spark.sparkContext.defaultParallelism}")
      val randomForest = new RandomForestCARTClassifier()
        .setMaxBins(32)
        .setMaxDepth(30)
        .setMinInstancesPerNode(1)
        .setNumTrees(500)
        .setSeed(123L)

      val model = randomForest.fit(train)
      // model.save(output)
      model
    })
    spark.stop()
  }
}
