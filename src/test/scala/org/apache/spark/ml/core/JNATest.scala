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

  def sumScala(array: Array[Double], n: Int): Double = {
    var sum: Double = 0
    var i = 0
    while ( i < n ) {
      sum += array(i)
      i += 1
    }
    sum
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

  def main(args: Array[String]): Unit = {
    testThreeCyclePerf()
  }
}

// scalastyle:on println

