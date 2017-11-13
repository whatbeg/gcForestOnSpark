/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package examples.UCI_adult

import scopt.OptionParser

object Utils {
  case class TrainParams(
              trainFile: String = "./data/uci_adult/sample_adult.data",
              testFile: String = "./data/uci_adult/sample_adult.test",
              featuresFile: String = "./data/uci_adult/features",
              model: String = "./models/uci_adult",
              classNum: Int = 2,
              multiScanWindow: Array[Int] = Array(),
              scanForestTreeNum: Int = 2,
              cascadeForestTreeNum: Int = 500,
              scanMinInsPerNode: Int = 2,
              cascadeMinInsPerNode: Int = 2,
              maxBins: Int = 32,
              maxDepth: Int = 30,
              maxIteration: Int = 10,
              maxMemoryInMB: Int = 2048,
              numFolds: Int = 3,
              earlyStoppingRounds: Int = 4,
              earlyStopByTest: Boolean = true,
              dataStyle: String = "Seq",
              dataSize: Array[Int] = Array(113),
              seed: Long = 123,
              cacheNodeId: Boolean = true,
              debugLevel: String = "ERROR",
              winCol: String = "windows",
              scanCol: String = "scan_id",
              forestIdCol: String = "forestNum",
              instanceCol: String = "instance",
              rawPredictionCol: String = "rawPrediction",
              probabilityCol: String = "probability",
              predictionCol: String = "prediction",
              featuresCol: String = "features",
              labelCol: String = "label",
              idebug: Boolean = false)

  val trainParser = new OptionParser[TrainParams]("GCForest On Spark - UCI ADULT Example") {
    head("Train Multi-grain Scan Cascade Forest for UCI ADULT")
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
    opt[String]("msWin")
      .text("Multi-grain Scan Window, an Array contains length(Seq) or width and height(Img), default: (), format: x,y")
      .action((x, c) => c.copy(multiScanWindow = x.split(",").map(num => num.toInt).toSeq.toArray))
    opt[Int]("scanTreeNum")
      .text("scanning Forest tree Number, default: 2")
      .action((x, c) => c.copy(scanForestTreeNum = x))
    opt[Int]("casTreeNum")
      .text("cascade Forest tree Number, default: 500")
      .action((x, c) => c.copy(cascadeForestTreeNum = x))
    opt[Int]("scanMinInsPerNode")
      .text("scaning Forest Minimum Instances per Node, default: 2")
      .action((x, c) => c.copy(scanMinInsPerNode = x))
    opt[Int]("casMinInsPerNode")
      .text("cascade Forest Minimum Instances per Node, default: 2")
      .action((x, c) => c.copy(cascadeMinInsPerNode = x))
    opt[Int]('b', "maxBins")
      .text("random Forest max Bins to split continuous features, default: 32")
      .action((x, c) => c.copy(maxBins = x))
    opt[Int]('d', "maxDepth")
      .text("random Forest max Depth, default: 30")
      .action((x, c) => c.copy(maxDepth = x))
    opt[Int]('i', "maxIteration")
      .text("max Iteration to grow cascade Forests, default: 10")
      .action((x, c) => c.copy(maxIteration = x))
    opt[Int]('m', "maxMemoryInMB")
      .text("max memory to histogram aggregates, default: 256")
      .action((x, c) => c.copy(maxMemoryInMB = x))
    opt[Int]("numFolds")
      .text("number of Cross Validation Folds to generator class vectors, default: 3")
      .action((x, c) => c.copy(numFolds = x))
    opt[Int]("esRound")
      .text("number of round to early stopping, default: 4")
      .action((x, c) => c.copy(earlyStoppingRounds = x))
    opt[Int]("esbTest")
      .text("whether conduct early stop by test metric, default: 1 (true), 0 represents false")
      .action((x, c) => c.copy(earlyStopByTest = if (x == 1) true else false))
    opt[String]("dStyle")
      .text("data Style to classify, Image or Seq, default: Seq")
      .action((x, c) => c.copy(dataStyle = x))
    opt[String]("dSize")
      .text("data Size to classify, an Array contains features dimension(Seq) " +
        "or width and height(Img), default: (113,), format: x,y")
      .action((x, c) => c.copy(dataSize = x.split(",").map(num => num.toInt).toSeq.toArray))
    opt[Long]("seed")
      .text("random seed, default: 123L")
      .action((x, c) => c.copy(seed = x))
    opt[String]("cacheNodeId")
      .text("cache node id or not, default: true")
      .action((x, c) => c.copy(cacheNodeId = x == "y"))
    opt[String]("debugLevel")
      .text("debug level you want to set, default: ERROR")
      .action((x, c) => c.copy(debugLevel = x))
    opt[String]("idebug")
      .text("if print debug infomation, default: n (y or n)")
      .action((x, c) => c.copy(idebug = x == "y"))
    // other parameters do not need to change
  }
}
