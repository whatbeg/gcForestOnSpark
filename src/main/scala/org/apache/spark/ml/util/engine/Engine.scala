/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.util.engine

import org.apache.spark.{SparkConf, SparkContext}

object Engine {

  private var nodeNum = 1
  private var coresPerNode = 1

  private def getTotalCores: Int = {
    // We assume the HT is enabled
    // Todo: check the Hyper threading, now we ignore HT
    Runtime.getRuntime.availableProcessors()
  }

  private def dynamicAllocationExecutor(conf: SparkConf): Option[Int] = {
    if (conf.get("spark.dynamicAllocation.enabled", null) == "true") {
      val maxExecutors = conf.get("spark.dynamicAllocation.maxExecutors", "1").toInt
      val minExecutors = conf.get("spark.dynamicAllocation.minExecutors", "1").toInt
      require(maxExecutors == minExecutors, "Engine.init: " +
        "spark.dynamicAllocation.maxExecutors and " +
        "spark.dynamicAllocation.minExecutors must be identical " +
        "in dynamic allocation for BigDL")
      Some(minExecutors)
    } else {
      None
    }
  }

  /**
    * Extract spark executor number and executor cores from given conf.
    * Exposed for testing.
    * @return (nExecutor, executorCore)
    */
  private[util] def parseExecutorAndCore(conf: SparkConf): Option[(Int, Int)] = {
    val master = conf.get("spark.master", null)
    if (master.toLowerCase.startsWith("local")) {
      // Spark local mode
      val patternLocalN = "local\\[(\\d+)\\]".r
      val patternLocalStar = "local\\[\\*\\]".r
      master match {
        case patternLocalN(n) => Some(1, n.toInt)
        case patternLocalStar(_*) => Some(1, getTotalCores)
        case _ => throw new IllegalArgumentException(s"Can't parser master $master")
      }
    } else if (master.toLowerCase.startsWith("spark")) {
      // Spark standalone mode
      val coreString = conf.get("spark.executor.cores", null)
      val maxString = conf.get("spark.cores.max", null)
      require(coreString != null, "Engine.init: Can't find executor core number" +
        ", do you submit with --executor-cores option")
      require(maxString != null, "Engine.init: Can't find total core number" +
        ". Do you submit with --total-executor-cores")
      val core = coreString.toInt
      val nodeNum = dynamicAllocationExecutor(conf).getOrElse {
        val total = maxString.toInt
        require(total >= core && total % core == 0, s"Engine.init: total core " +
          s"number($total) can't be divided " +
          s"by single core number($core) provided to spark-submit")
        total / core
      }
      Some(nodeNum, core)
    } else if (master.toLowerCase.startsWith("yarn")) {
      // yarn mode
      val coreString = conf.get("spark.executor.cores", null)
      require(coreString != null, "Engine.init: Can't find executor core number" +
        ", do you submit with " +
        "--executor-cores option")
      val core = coreString.toInt
      val node = dynamicAllocationExecutor(conf).getOrElse {
        val numExecutorString = conf.get("spark.executor.instances", null)
        require(numExecutorString != null, "Engine.init: Can't find executor number" +
          ", do you submit with " +
          "--num-executors option")
        numExecutorString.toInt
      }
      Some(node, core)
    } else if (master.toLowerCase.startsWith("mesos")) {
      // mesos mode
      require(conf.get("spark.mesos.coarse", null) != "false", "Engine.init: " +
        "Don't support mesos fine-grained mode")
      val coreString = conf.get("spark.executor.cores", null)
      require(coreString != null, "Engine.init: Can't find executor core number" +
        ", do you submit with --executor-cores option")
      val core = coreString.toInt
      val nodeNum = dynamicAllocationExecutor(conf).getOrElse {
        val maxString = conf.get("spark.cores.max", null)
        require(maxString != null, "Engine.init: Can't find total core number" +
          ". Do you submit with --total-executor-cores")
        val total = maxString.toInt
        require(total >= core && total % core == 0, s"Engine.init: total core " +
          s"number($total) can't be divided " +
          s"by single core number($core) provided to spark-submit")
        total / core
      }
      Some(nodeNum, core)
    } else {
      throw new IllegalArgumentException(s"Engine.init: Unsupported master format $master")
    }
  }

  /**
    * Reset engine envs. Test purpose
    */
  private[spark] def reset(): Unit = {
    nodeNum = 1
    coresPerNode = 1
  }

  def getParallelism(spark: SparkContext): Int = {
    val conf = parseExecutorAndCore(spark.getConf)
    conf match {
      case Some((n, c)) =>
        nodeNum = n
        coresPerNode = c
      case None =>
        nodeNum = 1
        coresPerNode = 1
    }
    nodeNum * coresPerNode
  }

}
