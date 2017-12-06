/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package org.apache.spark.ml.tree

import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.HasSeed


private[ml] trait GCForestParams extends HasSeed {

  final val classNum: IntParam = new IntParam(
    this, "numClasses", "", (value: Int) => value > 0)

  setDefault(classNum -> 2)
  def setNumClasses(value: Int): this.type = set(classNum, value)

  final val modelPath: Param[String] = new Param[String](this, "modelPath", "")
  setDefault(modelPath -> "./gcf_model")
  def setModelPath(value: String): this.type = set(modelPath, value)

  final val multiScanWindow: IntArrayParam = new IntArrayParam(
    this, "multiScanWindow", "", (value: Array[Int]) => value.length >= 0)

  setDefault(multiScanWindow -> Array[Int](7, 7))
  def setMultiScanWindow(value: Array[Int]): this.type = set(multiScanWindow, value)

  final val scanForestTreeNum: IntParam = new IntParam(
    this, "scanForestTreeNum", "", (value: Int) => value > 0)

  setDefault(scanForestTreeNum -> 4)
  def setScanForestTreeNum(value: Int): this.type = set(scanForestTreeNum, value)

  final val scanMinInsPerNode: IntParam = new IntParam(
    this, "scanForestMinInstancesPerNode", "", (value: Int) => value > 0)

  setDefault(scanMinInsPerNode -> 1)
  def setScanForestMinInstancesPerNode(value: Int): this.type =
    set(scanMinInsPerNode, value)

  final val MaxIteration: IntParam = new IntParam(
    this, "cascadeForestMaxIteration", "", (value: Int) => value > 0)

  setDefault(MaxIteration -> 4)
  def setMaxIteration(value: Int): this.type = set(MaxIteration, value)

  final val cascadeForestTreeNum: IntParam = new IntParam(
    this, "cascadeForestTreeNum", "", (value: Int) => value > 0)

  setDefault(cascadeForestTreeNum -> 4)
  def setCascadeForestTreeNum(value: Int): this.type = set(cascadeForestTreeNum, value)

  final val cascadeMinInsPerNode: IntParam = new IntParam(
    this, "cascadeForestMinInstancesPerNode", "", (value: Int) => value > 0)

  setDefault(cascadeMinInsPerNode -> 1)
  def setCascadeForestMinInstancesPerNode(value: Int): this.type =
    set(cascadeMinInsPerNode, value)

  final val featureSubsetStrategy: Param[String] = new Param[String](
    this, "featureSubsetStrategy", "")
  setDefault(featureSubsetStrategy -> "sqrt")

  final val crf_featureSubsetStrategy: Param[String] = new Param[String](
    this, "crf_featureSubsetStrategy", ""
  )
  setDefault(crf_featureSubsetStrategy -> "log2")

  def setFeatureSubsetStrategy(value: String): this.type = set(featureSubsetStrategy, value)

  def setCrf_featureSubsetStrategy(value: String): this.type = set(crf_featureSubsetStrategy, value)

  final val MaxBins: IntParam =
    new IntParam(this, "randomForestMaxBins", "", (value: Int) => value > 0)
  setDefault(MaxBins -> 32)
  def setMaxBins(value: Int): this.type = set(MaxBins, value)

  final val MaxDepth: IntParam =
    new IntParam(this, "randomForestMaxDepth", "", (value: Int) => value > 0)
  setDefault(MaxDepth -> 30)
  def setMaxDepth(value: Int): this.type = set(MaxDepth, value)

  final val minInfoGain: DoubleParam = new DoubleParam(this, "min Info Gain", "",
    (value: Double) => value >= 0)
  setDefault(minInfoGain -> 1e-6)
  def setMinInfoGain(value: Double): this.type = set(minInfoGain, value)

  final val numFolds: IntParam = new IntParam(this, "numFolds", "", (value: Int) => value > 0)

  setDefault(numFolds -> 3)
  def setNumFolds(value: Int): this.type = set(numFolds, value)

  final val earlyStoppingRounds: IntParam =
    new IntParam(this, "earlyStoppingRounds", "", (value: Int) => value > 0)
  setDefault(earlyStoppingRounds -> 4)
  def setEarlyStoppingRounds(value: Int): this.type = set(earlyStoppingRounds, value)

  final val earlyStopByTest: BooleanParam = new BooleanParam(this, "earlyStopByTest", "")
  setDefault(earlyStopByTest -> true)
  def setEarlyStopByTest(value: Boolean): this.type = set(earlyStopByTest, value)

  final val dataStyle: Param[String] = new Param[String](
    this, "dataStyle", "", (value: String) => Array("Seq", "Img").contains(value)) // TODO

  setDefault(dataStyle -> "Seq")
  def setDataStyle(value: String): this.type = set(dataStyle, value)

  val dataSize: IntArrayParam = new IntArrayParam(this, "dataSize", "",
    (value: Array[Int]) => value.length >= 0)

  def setDataSize(value: Array[Int]): this.type = set(dataSize, value)
  def getDataSize: Array[Int] = $(dataSize)

  final val rfNum: IntParam =
    new IntParam(this, "rfNum", "random forest num in cascade layer", (value: Int) => value >= 0)
  setDefault(rfNum -> 4)
  def setRFNum(value: Int): this.type = set(rfNum, value)

  final val crfNum: IntParam =
    new IntParam(this, "crfNum", "completely random forest num in cascade layer",
      (value: Int) => value >= 0)
  setDefault(crfNum -> 4)
  def setCRFNum(value: Int): this.type = set(crfNum, 4)

  final val windowCol: Param[String] = new Param[String](this, "windowCol", "windowId column name")
  setDefault(windowCol -> "window")

  final val scanCol: Param[String] = new Param[String](this, "scanCol", "scanId column name")
  setDefault(scanCol -> "scan_id")

  final val forestIdCol: Param[String] =
    new Param[String](this, "forestIdCol", "forest id column name")
  setDefault(forestIdCol -> "forestNum")

  final val idebug: Param[Boolean] = new Param[Boolean](this, "idebug", "if debug or not")
  setDefault(idebug -> false)

  def setIDebug(value: Boolean): this.type = set(idebug, value)

  def setSeed(value: Long): this.type = set(seed, value)

  final val cacheNodeId: Param[Boolean] = new Param[Boolean](
    this, "cacheNodeId", "if cache node id or not")
  setDefault(cacheNodeId -> true)

  def setCacheNodeId(value: Boolean): this.type = set(cacheNodeId, value)

  final val maxMemoryInMB: Param[Int] = new Param[Int](
    this, "maxMemoryInMB", "max memory to histogram aggregates")
  setDefault(maxMemoryInMB -> 256)

  def setMaxMemoryInMB(value: Int): this.type = set(maxMemoryInMB, value)
}
