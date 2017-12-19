/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package examples.RandomForest

import datasets.UCI_adult
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.gcForestEvaluator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession
import org.apache.spark.util.SizeEstimator
import org.apache.spark.ml.util.engine.Engine

object DecisionTreeExample {
  def main(args: Array[String]): Unit = {

    import Utils._

    val spark = SparkSession.builder()
      .appName(this.getClass.getSimpleName)
//      .master("local[*]")
      .getOrCreate()

    val parallelism = Engine.getParallelism(spark.sparkContext)
    println(s"Total Cores is $parallelism")
    //    spark.conf.set("spark.locality.wait.node", 0)

    trainParser.parse(args, TrainParams()).foreach(param => {

      spark.sparkContext.setLogLevel(param.debugLevel)
      //      spark.sparkContext.setCheckpointDir("./checkpoint")
      // if param set to negative, use dataRDD
      // else if param set to positive, use repartition(param.parallelism)
      // else if param set to 0, use Engine.parallelism
      def getParallelism: Int = param.parallelism match {
        case p if p > 0 => param.parallelism
        case n if n < 0 => -1
        case _ => parallelism
      }
      val train = new UCI_adult().load_data(spark, param.trainFile, param.featuresFile, 1,
        getParallelism)
      val test = new UCI_adult().load_data(spark, param.testFile, param.featuresFile, 1,
        getParallelism)

      println(s"Train set shape (${train.count()}, ${train.head.getAs[Vector]("features").size})")

      val stime = System.currentTimeMillis()
      val decision_tree = new DecisionTreeClassifier()
        .setMaxBins(param.maxBins)
        .setMaxDepth(param.maxDepth)
        .setMinInstancesPerNode(param.MinInsPerNode)
        .setMaxMemoryInMB(param.maxMemoryInMB)
        .setMinInfoGain(param.minInfoGain)
        .setSeed(param.seed)
        .setCacheNodeIds(param.cacheNodeId)

      val model = decision_tree.fit(train)
      println("Model Size estimates: %.1f M".format(SizeEstimator.estimate(model) / 1048576.0))
      println(s"Fit a random forest in Spark cost ${(System.currentTimeMillis() - stime) / 1000.0} s")
      println(s"Total nodes: ${model.numNodes}")

      val predictions = model.transform(test)

      // Select example rows to display.
      predictions.select("probability", "label", "features").show(5)
      val accuracy = gcForestEvaluator.evaluate(predictions.withColumnRenamed("probability", "features"))

      println(s"[$getNowTime] Test Accuracy = " + accuracy)
      model
    })
    spark.stop()
  }
}
