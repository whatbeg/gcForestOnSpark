/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package org.apache.spark.ml.param.shared

import org.apache.spark.ml.param.{Param, Params}


private[ml] trait HasInstanceCol extends Params {
  final val instanceCol: Param[String] =
    new Param[String](this, "instanceCol", "instance number column name")

  setDefault(instanceCol, "instance")

  /** @group getParam */
  final def getInstanceCol: String = $(instanceCol)
}
