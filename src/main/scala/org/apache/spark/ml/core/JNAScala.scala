/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.core

import com.sun.jna._

// scalastyle:off println

object JNAScala {
//  Native.register("test")
  Native.register("binToBestSplit")
//  @native def add(a: Int, b: Int): Int
//  @native def sum(arr: Array[Double], n: Int): Double
//  @native def getSomeContentsForPointer(size: Long): Pointer

  @native def calculateImpurity(
              impurity: Char,
              allStats: Array[Double],
              statSize: Int,
              offset: Int): Double
  @native def calcGainAndImpurityStats(impurity: Char,
                                       ImpurityStats: Array[Double],
                                       statSize: Int,
                                       numSplits: Int,
                                       allStats: Array[Double],
                                       allStatsSize: Int,
                                       nodeFeatureOffset: Int,
                                       leftOffset: Int,
                                       minInsPerNode: Int,
                                       minInfoGain: Double): Pointer

  @native def binToBestSplit(ImpurityStats: Array[Double],
                             allStats: Array[Double],
                             featureOffset: Array[Int],
                             nfeatureOffset: Int,
                             numSplits: Int,
                             impurity: Char,
                             statSize: Int,
                             featureIndexIdx: Int,
                             minInsPerNode: Int,
                             minInfoGain: Double): Pointer

  def sumScala(array: Array[Double], n: Int): Double = {
    var sum: Double = 0
    var i = 0
    while ( i < n ) {
      sum += array(i)
      i += 1
    }
    sum
  }

  def scalaCalculateImpurity(impurity: Char, allStats: Array[Double], statSize: Int, offset: Int):
  Double = {
    if (impurity == 'g') { // gini
      val totalCount = Range(offset, offset + statSize).map(allStats(_)).sum
      if (totalCount <= 1e-9) {
        0.0
      }
      else {
        var impurityResult = 1.0
//        println(s"total count = $totalCount")
        var i = 0
        while (i < statSize) {
          val freq = allStats(offset + i) / totalCount
//          println(s"freq = $freq")
          impurityResult -= freq * freq
          i += 1
        }
        impurityResult
      }
    }
    else if (impurity == 'e') { // entropy
      -1.0
    }
    else if (impurity == 'v') { // variance
      -1.0
    } else -1.0
  }

  def testCalculateImpurity(): Unit = {
    val impurity = 'g'
    val allStats = Array(1.0, 1.0, 12.0, 2.0, 14.0, 3.0)
    val statSize = 2
    val offset = 2
    val res = calculateImpurity(impurity, allStats, statSize, offset)
    println("Native Res = " + res)
    val scalaRes = scalaCalculateImpurity(impurity, allStats, statSize, offset)
    println("scala Res = " + scalaRes)
    assert(res == scalaRes)
  }

  def testCalcGainAndImpurityStats(): Unit = {
    val allStats = Array(1.0, 1, 12, 2, 14, 3)
    val ImpurityStats = Array(0.0, 0, 0, 0, 0, 0, 0, 0, -1);  // stats = null
    val p = calcGainAndImpurityStats('g', ImpurityStats, 2, 2, allStats, 6, 0,
      0, 2, 1e-7)
    println(p.getDoubleArray(0, 3 + 2 * 3).mkString(","))
    Native.free(Pointer.nativeValue(p))
    Pointer.nativeValue(p, 0)
  }

  def testBinToBestSplit(): Unit = {
    val allStats = Array(1.0, 1.0, 11.0, 1.0, 2.0, 1.0)
    val ImpurityStats = Array(0.0, 0.0, 0, 0, 0, 0, 0, 0, -1);  // stats = null
    val featureOffset = Array[Int](0, 6)
    val p = binToBestSplit(ImpurityStats, allStats, featureOffset, 2, 2, 'g', 2, 0, 2, 1e-8)
    println(p.getDoubleArray(0, 4 + 2 * 3).mkString(","))
    Native.free(Pointer.nativeValue(p))
    Pointer.nativeValue(p, 0)
  }

//  def main(args: Array[String]): Unit = {
//    testCalculateImpurity()
//    testCalcGainAndImpurityStats()
//    testBinToBestSplit()
//  }
}

// scalastyle:on println

//    println(add(10, 20))
//    val arr = Array.fill[Double](10000)(1d)
//    val start_time = System.nanoTime()
//    sum(arr, 10000)
//    val ctime = System.nanoTime() - start_time
//    println(s"JNA cost time: ${ctime / 1e6} ms")
//    val stime = System.nanoTime()
//    sumScala(arr, 10000)
//    val etime = System.nanoTime() - stime
//    println(s"Scala cost time: ${etime / 1e6} ms")
