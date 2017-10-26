/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package datasets

import org.scalatest.FlatSpec

class UCI_adultSpec extends FlatSpec {

  val data = new UCI_adult().load_data("train", 1)
  val list = data.collectAsList()
  println(list)
}
