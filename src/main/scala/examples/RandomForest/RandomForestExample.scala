/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package examples.RandomForest

import datasets.UCI_adult
import org.apache.spark.ml.classification.{RandomForestCARTClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.utils.engine.Engine

object RandomForestExample {
  def main(args: Array[String]): Unit = {

    import Utils._

    val spark = SparkSession.builder()
      .appName(this.getClass.getSimpleName)
      .master("local[*]")
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

      val train = new UCI_adult().load_data(spark, param.trainFile, param.featuresFile, 1, parallelism)
      val test = new UCI_adult().load_data(spark, param.testFile, param.featuresFile, 1, parallelism)

      val randomForest = new RandomForestClassifier()
        .setMaxBins(param.maxBins)
        .setMaxDepth(param.maxDepth)
        .setMinInstancesPerNode(param.MinInsPerNode)
        .setNumTrees(param.ForestTreeNum)
        .setSeed(param.seed)

      val model = randomForest.fit(train)

      val predictions = model.transform(test)

      // Select example rows to display.
      predictions.select("prediction", "label", "features").show(5)
      val accuracy = Evaluator.evaluatePrediction(predictions)

      println("Test Accuracy = " + accuracy)
      if (param.idebug) println("Learned classification GBT model:\n" + model.toDebugString)

      model
    })
    spark.stop()
  }
}
