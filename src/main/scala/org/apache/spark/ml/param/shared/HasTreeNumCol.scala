/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package org.apache.spark.ml.param.shared

import org.apache.spark.ml.param.{Param, Params}


private[ml] trait HasTreeNumCol extends Params {
  final val forestNumCol: Param[String] =
    new Param[String](this, "forestNumCol", "forest number column name")

  setDefault(forestNumCol, "forestNum")

  /** @group getParam */
  final def getTreeNumCol: String = $(forestNumCol)
}


