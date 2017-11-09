/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package examples.RandomForest

import datasets.UCI_adult
import org.apache.spark.ml.classification.CompletelyRandomForestClassifier
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.utils.engine.Engine

object CompletelyRandomForestExample {
  def main(args: Array[String]): Unit = {

    import Utils._

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

      val train = new UCI_adult().load_data(spark, param.trainFile, param.featuresFile, 1,
        if (param.parallelism > 0) param.parallelism else parallelism)
      val test = new UCI_adult().load_data(spark, param.testFile, param.featuresFile, 1,
        if (param.parallelism > 0) param.parallelism else parallelism)

      val randomForest = new CompletelyRandomForestClassifier()
        .setMaxBins(param.maxBins)
        .setMaxDepth(param.maxDepth)
        .setMinInstancesPerNode(param.MinInsPerNode)
        .setNumTrees(param.ForestTreeNum)
        .setSeed(param.seed)
        .setCacheNodeIds(param.cacheNodeId)

      val model = randomForest.fit(train)

      val predictions = model.transform(test)

      // Select example rows to display.
      predictions.select("probability", "label", "features").show(5)
      val accuracy = Evaluator.evaluate(predictions.withColumnRenamed("probability", "features"))

      println(s"[${getNowTime}] Test Accuracy = " + accuracy)
      if (param.idebug) println("Learned classification Random Forest model:\n" + model.toDebugString)

      model
    })
    spark.stop()
  }
}
