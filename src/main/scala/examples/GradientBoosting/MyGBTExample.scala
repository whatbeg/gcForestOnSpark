/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package examples.GradientBoosting

import datasets.UCI_adult
import org.apache.spark.ml.classification.GradientBoostingTreeClassifier
import org.apache.spark.ml.evaluation.gcForestEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.util.engine.Engine

object MyGBTExample {
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

      val trainingData = new UCI_adult().load_data(spark, param.trainFile, param.featuresFile, 1,
        if (param.parallelism > 0) param.parallelism else parallelism)
      val testData = new UCI_adult().load_data(spark, param.testFile, param.featuresFile, 1,
        if (param.parallelism > 0) param.parallelism else parallelism)

      val gbt = new GradientBoostingTreeClassifier()
        .setMaxIter(param.numIteration)
        .setMaxBins(param.maxBins)
        .setLabelCol("label")
        .setFeaturesCol("features")

      val model = gbt.fit(trainingData)
      // Evaluate model on test instances and compute test error
      val predictions = model.transform(testData)
      predictions.select("probability", "label", "features").show(5)

      val accuracy = gcForestEvaluator.evaluate(predictions.withColumnRenamed("probability", "features"))
      println(s"[$getNowTime] Test Accuracy = " + accuracy)
      if (param.idebug) println("Learned classification GBT model:\n" + model.toDebugString)
      if (param.idebug) println("Total Num nodes: " + model.totalNumNodes)
      if (param.idebug) println("Total Trees: " + model.getNumTrees)
      model
    })
    spark.stop()
  }
}
