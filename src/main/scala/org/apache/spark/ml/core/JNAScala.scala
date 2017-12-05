/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.core

import com.sun.jna._

object JNAScala {

  Native.register("binToBestSplit")

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

}
