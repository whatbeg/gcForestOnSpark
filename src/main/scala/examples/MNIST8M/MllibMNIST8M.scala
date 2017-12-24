/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package examples.MNIST8M

import org.apache.spark.ml.classification.{DecisionTreeClassifier, YggdrasilClassifier}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession
import org.apache.spark.util.SizeEstimator

object MllibMNIST8M {
  def main(args: Array[String]): Unit = {

    import Utils._

    val spark = SparkSession.builder()
      .appName(this.getClass.getSimpleName)
//      .master("local[*]")
      .getOrCreate()

    trainParser.parse(args, TrainParams()).foreach(param => {

      spark.sparkContext.setLogLevel(param.debugLevel)

      val train = spark.read.format("libsvm").load(param.trainFile)

      println(s"Train set shape (${train.count()}, ${train.head.getAs[Vector]("features").size})")

      train.show(5)

      val stime = System.currentTimeMillis()
      val mllib_tree = new DecisionTreeClassifier()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setMaxDepth(param.maxDepth)
        .setMaxBins(param.maxBins)
        .setMinInstancesPerNode(param.MinInsPerNode)
        .setMinInfoGain(param.minInfoGain)
        .setCacheNodeIds(param.cacheNodeId)

      val model = mllib_tree.fit(train)
      println(s"Fit a MLlib decision tree in Spark cost ${(System.currentTimeMillis() - stime) / 1000.0} s")
      println("Model Size estimates: %.1f M".format(SizeEstimator.estimate(model) / 1048576.0))
      println(s"Total nodes: ${model.numNodes}")

//      val predictions = model.transform(test)
//
//      // Select example rows to display.
//      predictions.select("probability", "label", "features").show(5)
//      val accuracy = gcForestEvaluator.evaluate(predictions.withColumnRenamed("probability", "features"))
//
//      println(s"[$getNowTime] Test Accuracy = " + accuracy)
      model
    })
    spark.stop()
  }
}
