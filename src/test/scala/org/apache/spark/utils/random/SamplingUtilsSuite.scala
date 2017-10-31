/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.utils.random

import org.apache.spark.ml.Utils.SparkUnitTest

import scala.util.Random

class SamplingUtilsSuite extends SparkUnitTest {
  test("Sample Generator") {
    val array = Range(0, 10)
    val random = new Random()
    random.setSeed(123)
    val disdinctSampled = Range(0, 4).map { r =>
      val sampled = SamplingUtils.reservoirSampleAndCount(array.toIterator, 4, random.nextLong())
      // println(sampled._1.toSeq.mkString)
      sampled._1.toSeq.mkString(",")
    }
    assert(disdinctSampled.distinct.length == disdinctSampled.length)
  }
}
