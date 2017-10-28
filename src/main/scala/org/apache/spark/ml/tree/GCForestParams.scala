/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package org.apache.spark.ml.tree

import org.apache.spark.ml.param.{BooleanParam, IntArrayParam, IntParam, Param}
import org.apache.spark.ml.param.shared.{HasSeed, HasTreeNumCol}


private[ml] trait GCForestParams extends HasSeed with HasTreeNumCol {

  final val multiScanWindow: IntArrayParam = new IntArrayParam(
    this, "multiScanWindow", "", (value: Array[Int]) => value.length >= 0)

  setDefault(multiScanWindow -> Array[Int](7, 7))
  def setMultiScanWindow(value: Array[Int]): this.type = set(multiScanWindow, value)

  final val scanForestTreeNum: IntParam = new IntParam(
    this, "scanForestTreeNum", "", (value: Int) => value > 0)

  setDefault(scanForestTreeNum -> 4)
  def setScanForestTreeNum(value: Int): this.type = set(scanForestTreeNum, value)

  final val scanForestMinInstancesPerNode: IntParam = new IntParam(
    this, "scanForestMinInstancesPerNode", "", (value: Int) => value > 0)

  setDefault(scanForestMinInstancesPerNode -> 1)
  def setScanForestMinInstancesPerNode(value: Int): this.type =
    set(scanForestMinInstancesPerNode, value)

  final val cascadeForestMaxIteration: IntParam = new IntParam(
    this, "cascadeForestMaxIteration", "", (value: Int) => value > 0)

  setDefault(cascadeForestMaxIteration -> 4)
  def setMaxIteration(value: Int): this.type = set(cascadeForestMaxIteration, value)

  final val cascadeForestTreeNum: IntParam = new IntParam(
    this, "cascadeForestTreeNum", "", (value: Int) => value > 0)

  setDefault(cascadeForestTreeNum -> 4)
  def setCascadeForestTreeNum(value: Int): this.type = set(cascadeForestTreeNum, value)

  final val cascadeForestMinInstancesPerNode: IntParam = new IntParam(
    this, "cascadeForestMinInstancesPerNode", "", (value: Int) => value > 0)

  setDefault(cascadeForestMinInstancesPerNode -> 1)
  def setCascadeForestMinInstancesPerNode(value: Int): this.type =
    set(cascadeForestMinInstancesPerNode, value)

  final val randomForestMaxBins: IntParam =
    new IntParam(this, "randomForestMaxBins", "", (value: Int) => value > 0)
  setDefault(randomForestMaxBins -> 32)
  def setRandomForestMaxBins(value: Int): this.type = set(randomForestMaxBins, value)

  final val randomForestMaxDepth: IntParam =
    new IntParam(this, "randomForestMaxDepth", "", (value: Int) => value > 0)
  setDefault(randomForestMaxDepth -> 30)
  def setRandomForestMaxDepth(value: Int): this.type = set(randomForestMaxDepth, value)

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
    this, "dataStyle", "", (value: String) => Array("sequence", "image").contains(value)) // TODO

  setDefault(dataStyle -> "image")
  def setDataStyle(value: String): this.type = set(dataStyle, value)

  val dataSize: IntArrayParam = new IntArrayParam(this, "dataSize", "") // TODO

  def setDataSize(value: Array[Int]): this.type = set(dataSize, value)
  def getDataSize: Array[Int] = $(dataSize)

  final val instanceCol: Param[String] =
    new Param[String](this, "instanceCol", "instanceId column name")
  setDefault(instanceCol -> "instance")

  final val windowCol: Param[String] = new Param[String](this, "windowCol", "windowId column name")
  setDefault(windowCol -> "window")

  final val scanCol: Param[String] = new Param[String](this, "scanCol", "scanId column name")
  setDefault(scanCol -> "scan_id")

  def setSeed(value: Long): this.type = set(seed, value)
}
