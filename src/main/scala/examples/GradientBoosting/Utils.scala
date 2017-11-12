/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package examples.GradientBoosting

import java.text.SimpleDateFormat
import java.util.Date

import scopt.OptionParser

object Utils {
  val dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss,SSS")
  def getNowTime = dateFormat.format(new Date())

  case class TrainParams(
                          trainFile: String = "./data/uci_adult/adult_2000.data",
                          testFile: String = "./data/uci_adult/sample_adult.test",
                          featuresFile: String = "./data/uci_adult/features",
                          model: String = "./models/uci_adult",
                          classNum: Int = 2,
                          maxDepth: Int = 30,
                          maxBins: Int = 32,
                          numIteration: Int = 10,
                          seed: Long = 123,
                          debugLevel: String = "ERROR",
                          idebug: Boolean = false,
                          parallelism: Int = 0)

  val trainParser = new OptionParser[TrainParams]("GBT On Spark - UCI ADULT Example") {
    head("Train Gradient Boosting Tree for UCI ADULT")
    opt[String]("train")
      .text("where you put your training files, default: ./data/uci_adult/sample_adult.data")
      .action((x, c) => c.copy(trainFile = x))
    opt[String]("test")
      .text("where you put your testing files, default: ./data/uci_adult/sample_adult.test")
      .action((x, c) => c.copy(testFile = x))
    opt[String]("features")
      .text("where you put your features files, default: ./data/uci_adult/sample_adult.test")
      .action((x, c) => c.copy(featuresFile = x))
    opt[String]("model")
      .text("where you put your trained model, default: ./models/uci_adult")
      .action((x, c) => c.copy(model = x))
    opt[Int]("classNum")
      .text("number of Classes, default: 2")
      .action((x, c) => c.copy(classNum = x))
    opt[Int]('b', "maxBins")
      .text("random Forest max Bins to split continuous features, default: 32")
      .action((x, c) => c.copy(maxBins = x))
    opt[Int]('d', "maxDepth")
      .text("random Forest max Depth, default: 30")
      .action((x, c) => c.copy(maxDepth = x))
    opt[Int]('i', "numIteration")
      .text("max Iteration to grow cascade Forests, default: 10")
      .action((x, c) => c.copy(numIteration = x))
    opt[Long]("seed")
      .text("random seed, default: 123L")
      .action((x, c) => c.copy(seed = x))
    opt[String]("debugLevel")
      .text("debug level you want to set, default: ERROR")
      .action((x, c) => c.copy(debugLevel = x))
    opt[String]("idebug")
      .text("if print debug infomation, default: n (y or n)")
      .action((x, c) => c.copy(idebug = x == "y"))
    opt[Int]('p', "parallelism")
      .text("parallelism you want to set, default: 0")
      .action((x, c) => c.copy(parallelism = x))
    // other parameters do not need to change
  }
}
