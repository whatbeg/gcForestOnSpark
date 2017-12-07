/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.core

import com.sun.jna._
import core.JNITest

// scalastyle:off println

object JNATest {
  Native.register("test")
  //  Native.register("binToBestSplit")
  @native def sum(arr: Array[Double], n: Int): Double
  @native def calc(arr: Array[Double], n: Int): Double
  @native def calcIntensive(a: Double, n: Int): Double

  def sumScala(array: Array[Double], n: Int): Double = {
    var sum: Double = 0
    var i = 0
    while ( i < n ) {
      sum += array(i)
      i += 1
    }
    sum
  }
  
  def calcScala(arr: Array[Double], n: Int): Double = {
    var mess = 1.0
    for (i <- 0 until n) {
      mess = mess + (2 * arr(i) + 3 * arr(i) + 4 * arr(i)) /
        (5 * arr(i) + 1.0 + 2 * (arr(i) - 1 + 2.4 + 1.0 * 3.0 / 1.0))
      mess = mess + mess - mess * mess + mess / mess + mess * mess
      mess *= 1.0
      if (mess > 10000000000.0)
        mess /= 10000000000.0
    }
    mess
  }

  def calcIntensiveScala(a: Double, n: Int): Double = {
    var res = 1.0
    for (i <- 0 until n) {
      res = res * 1.0 * 2.0 * 3.0 * 4.0 / 6.0 / 2.0 / 2.0 + res * 1.0 * 2.0 * 3.0 * 4.0 / 6.0 / 2.0 / 2.0
      res = res * 10.0 * 20.0 * 30.0 * 40.0 / 60.0 / 20.0 / 20.0 + res * 10.0 * 20.0 * 30.0 * 40.0 / 60.0 / 20.0 / 20.0
      res = res * i - res * i + res
      res = res * 111111.0 / 111111.0
      res = res / 10000.0 * 10000.0
      res = res * 107070707 / 107070707
    }
    res
  }

  def testThreeCyclePerf(): Unit = {
    val JNI = new JNITest()
    Range(1, 2000000, 20000).foreach { len =>
      val arr = Array.fill[Double](len)(1d)
      val start_time = System.nanoTime()
      sum(arr, len)
      val ctime = System.nanoTime() - start_time
      // println(s"JNA cost time: ${ctime / 1e6} ms")
      val stime = System.nanoTime()
      sumScala(arr, len)
      val etime = System.nanoTime() - stime
      // println(s"Scala cost time: ${etime / 1e6} ms")
      val sstime = System.nanoTime()
      JNI.sum(arr, len)
      val eetime = System.nanoTime() - sstime
      // println(s"JNI cost time: ${eetime / 1e6} ms")
      val jstime = System.nanoTime()
      JNI.sumJava(arr, len)
      val jttime = System.nanoTime() - jstime
      println(s"${ctime / 1e6}  ${etime / 1e6}  ${eetime / 1e6} ${jttime / 1e6}")
    }
  }

  def testCalc(): Unit = {
    val JNI = new JNITest()
    Range(1, 2000000, 20000).foreach { len =>
      val arr = Array.fill[Double](len)(1d)
      val start_time = System.nanoTime()
      calc(arr, len)
      val ctime = System.nanoTime() - start_time

      val stime = System.nanoTime()
      calcScala(arr, len)
      val etime = System.nanoTime() - stime

      val sstime = System.nanoTime()
      JNI.calc(arr, len)
      val eetime = System.nanoTime() - sstime

      val jstime = System.nanoTime()
      JNI.calcJava(arr, len)
      val jttime = System.nanoTime() - jstime
      println(s"${ctime / 1e6}  ${etime / 1e6}  ${eetime / 1e6} ${jttime / 1e6}")
    }
  }

  def testCalcIntensive(): Unit = {
    val JNI = new JNITest()
    Range(1000000, 40000000, 1000000).foreach { i =>
      val start_time = System.nanoTime()
      calcIntensive(1.0, i)
      val ctime = System.nanoTime() - start_time

      val stime = System.nanoTime()
      calcIntensiveScala(1.0, i)
      val etime = System.nanoTime() - stime

      val sstime = System.nanoTime()
      JNI.calcIntensive(1.0, i)
      val eetime = System.nanoTime() - sstime

      val jstime = System.nanoTime()
      JNI.calcIntensiveJava(1.0, i)
      val jttime = System.nanoTime() - jstime
      println(s"${i / 1000000} ${ctime / 1e6}  ${etime / 1e6}  ${eetime / 1e6} ${jttime / 1e6}")
    }
  }

  def main(args: Array[String]): Unit = {
    // testThreeCyclePerf()
    //testCalc()
    testCalcIntensive()
  }
}

// scalastyle:on println

