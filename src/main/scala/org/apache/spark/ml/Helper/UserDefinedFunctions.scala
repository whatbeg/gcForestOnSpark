/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.Helper

import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.sql.functions.udf

private[spark] object UserDefinedFunctions {
  private[spark] def mergeVectorForKfold(k: Int) = k match {
    case 2 =>
      udf { (v1: Vector, v2: Vector) =>
        val sumVector = Array.fill[Double](v1.size)(0d)
        for (fsk <- Array(v1, v2)) {
          for (i <- sumVector.indices)
            sumVector(i) += fsk(i) / k
        }
        new DenseVector(sumVector)
      }
    case 3 =>
      udf { (v1: Vector, v2: Vector, v3: Vector) =>
        val sumVector = Array.fill[Double](v1.size)(0d)
        for (fsk <- Array(v1, v2, v3)) {
          for (i <- sumVector.indices)
            sumVector(i) += fsk(i) / k
        }
        new DenseVector(sumVector)
      }
  }
}
