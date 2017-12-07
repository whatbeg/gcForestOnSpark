/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package examples.RandomForest

import org.apache.spark.ml.datasets.UCI_adult
import org.apache.spark.ml.classification.RandomForestCARTClassifier
import org.apache.spark.ml.evaluation.gcForestEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.util.engine.Engine
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.util.SizeEstimator

object RandomForestExampleWithML {
  def main(args: Array[String]): Unit = {

    import Utils._

    val spark = SparkSession.builder()
      .appName(this.getClass.getSimpleName)
      .master("local[*]")
      .getOrCreate()

    trainParser.parse(args, TrainParams()).foreach(param => {

      val train = new UCI_adult().load_data(spark, param.trainFile, param.featuresFile, false)
      val test = new UCI_adult().load_data(spark, param.testFile, param.featuresFile, false)

      train.printSchema()
      train.schema.fields.foreach { k =>
        println(k.metadata)
      }
      train.show(20)

//      train.filter ( case (row: Row) =>
//          row.getAs[Vector]("features").toArray(9) > 1
//      )

      spark.sparkContext.setLogLevel(param.debugLevel)
//      //      spark.sparkContext.setCheckpointDir("./checkpoint")
//      // if param set to negative, use dataRDD
//      // else if param set to positive, use repartition(param.parallelism)
//      // else if param set to 0, use Engine.parallelism
//      def getParallelism: Int = param.parallelism match {
//        case p if p > 0 => param.parallelism
//        case n if n < 0 => -1
//        case _ => parallelism
//      }
//      println(s"Estimate trainset ${SizeEstimator.estimate(train)}, testset: ${SizeEstimator.estimate(test)}")
//      println(s"Train set shape (${train.count()}, ${train.head.getAs[Vector]("features").size-2})")
//
      Range(0, param.count).foreach { i =>
        val stime = System.currentTimeMillis()
        val randomForest = new RandomForestCARTClassifier()
          .setMaxBins(param.maxBins)
          .setMaxDepth(param.maxDepth)
          .setMinInstancesPerNode(param.MinInsPerNode)
          .setFeatureSubsetStrategy(param.featureSubsetStrategy)
          .setMaxMemoryInMB(param.maxMemoryInMB)
          .setMinInfoGain(param.minInfoGain)
          .setNumTrees(param.ForestTreeNum)
          .setSeed(param.seed * i)
          .setCacheNodeIds(param.cacheNodeId)

        val model = randomForest.fit(train)
        val predictions = model.transform(test)

        // Select example rows to display.
        predictions.select("probability", "label", "features").show(5)
        val accuracy = gcForestEvaluator.evaluate(predictions.withColumnRenamed("probability", "features"))

        println(s"[$getNowTime] Test Accuracy = " + accuracy)

        println("Model Size estimates: %.1f M".format(SizeEstimator.estimate(model) / 1048576.0))
        println(s"Fit a random forest in Spark cost ${(System.currentTimeMillis() - stime) / 1000.0} s")
      }
    })
    spark.stop()
  }
}
