/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package examples.RandomForest

import datasets.UCI_adult
import org.apache.spark.ml.classification.RandomForestCARTClassifier
import org.apache.spark.ml.evaluation.gcForestEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.util.SizeEstimator
import org.apache.spark.ml.util.engine.Engine

object RandomForestExample {
  def main(args: Array[String]): Unit = {

    import Utils._

    val spark = SparkSession.builder()
      .appName(this.getClass.getSimpleName)
//      .master("local[*]")
      .getOrCreate()

    val parallelism = Engine.getParallelism(spark.sparkContext)
    println(s"Create Spark Context Succeed! Parallelism is $parallelism")
    spark.conf.set("spark.default.parallelism", parallelism)
    spark.conf.set("spark.locality.wait.node", 0)

    trainParser.parse(args, TrainParams()).foreach(param => {

      spark.sparkContext.setLogLevel(param.debugLevel)
//      spark.sparkContext.setCheckpointDir("./checkpoint")

      val train = new UCI_adult().load_data(spark, param.trainFile, param.featuresFile, 1,
        if (param.parallelism > 0) param.parallelism else parallelism)
      val test = new UCI_adult().load_data(spark, param.testFile, param.featuresFile, 1,
        if (param.parallelism > 0) param.parallelism else parallelism)

      println(s"Estimate trainset ${SizeEstimator.estimate(train)}, testset: ${SizeEstimator.estimate(test)}")

      Range(0, param.count).foreach { _ =>
        val stime = System.currentTimeMillis()
        val randomForest = new RandomForestCARTClassifier()
          .setMaxBins(param.maxBins)
          .setMaxDepth(param.maxDepth)
          .setMinInstancesPerNode(param.MinInsPerNode)
          .setMinInfoGain(param.minInfoGain)
          .setNumTrees(param.ForestTreeNum)
          .setSeed(param.seed)
          .setCacheNodeIds(param.cacheNodeId)

        val model = randomForest.fit(train)
        println("Model Size estimates: %.1f M".format(SizeEstimator.estimate(model) / 1048576.0))
        println(s"Fit a random forest in Spark cost ${(System.currentTimeMillis() - stime) / 1000.0} s")
        Thread.sleep(60 * 1000)
      }

      println("Training End, Sleep 20 seconds")
      Thread.sleep(20 * 1000)

//      val model = models(0)
//      val predictions = model.transform(test)
//
//      // Select example rows to display.
//      predictions.select("probability", "label", "features").show(5)
//      val accuracy = gcForestEvaluator.evaluate(predictions.withColumnRenamed("probability", "features"))
//
//      println(s"[$getNowTime] Test Accuracy = " + accuracy)
//      model
    })
    spark.stop()
  }
}
