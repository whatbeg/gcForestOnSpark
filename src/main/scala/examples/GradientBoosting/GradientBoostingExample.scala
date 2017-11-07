/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package examples.GradientBoosting

import datasets.UCI_adult
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.{Evaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.sql.SparkSession
import org.apache.spark.utils.engine.Engine

object GradientBoostingExample {
  def main(args: Array[String]): Unit = {

    import Utils._

    val spark = SparkSession.builder()
      .appName(this.getClass.getSimpleName)
//            .master("local[*]")
      .getOrCreate()

    val parallelism = Engine.getParallelism(spark.sparkContext)
    println(s"Create Spark Context Succeed! Parallelism is $parallelism")
    spark.conf.set("spark.default.parallelism", parallelism)
    spark.conf.set("spark.locality.wait.node", 0)

    trainParser.parse(args, TrainParams()).map(param => {

      spark.sparkContext.setLogLevel(param.debugLevel)

      val trainingData = new UCI_adult().load_data(spark, param.trainFile, param.featuresFile, 1, parallelism)
      //        .repartition(parallelism)
      val testData = new UCI_adult().load_data(spark, param.testFile, param.featuresFile, 1, parallelism)
      //        .repartition(parallelism)

      val gbt = new GBTClassifier()
        .setMaxIter(param.numIteration)
        .setMaxBins(param.maxBins)
        .setLabelCol("label")
        .setFeaturesCol("features")

      val model = gbt.fit(trainingData)
      // Evaluate model on test instances and compute test error
      val predictions = model.transform(testData)
      predictions.select("prediction", "label", "features").show(5)

      val accuracy = Evaluator.evaluatePrediction(predictions)
      println("Test Accuracy = " + accuracy)
      if (param.idebug) println("Learned classification GBT model:\n" + model.toDebugString)
      model
    })
    spark.stop()
  }
}
