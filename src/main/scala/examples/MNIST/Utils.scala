/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package examples.MNIST

import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, FileSystem, Path}
import org.apache.hadoop.io.IOUtils


object Utils {
  val trainMean = 0.13066047740239506
  val trainStd = 0.3081078

  val testMean = 0.13251460696903547
  val testStd = 0.31048024

  /**
    * load binary file from HDFS
    * @param fileName
    * @return
    */
  def readHdfsByte(fileName: String): Array[Byte] = {
    val src: Path = new Path(fileName)
    var fs: FileSystem = null
    var in: FSDataInputStream = null
    try {
      fs = src.getFileSystem(new Configuration())
      in = fs.open(src)
      val byteArrayOut = new ByteArrayOutputStream()
      IOUtils.copyBytes(in, byteArrayOut, 1024, true)
      byteArrayOut.toByteArray
    } finally {
      if (null != in) in.close()
      if (null != fs) fs.close()
    }
  }

  /**
    * load mnist data.
    * read mnist from hdfs if data folder starts with "hdfs:", otherwise form local file.
    * @param featureFile
    * @param labelFile
    * @return
    */
  def load(featureFile: String, labelFile: String): (Array[Array[Double]], Array[Double]) = {

    val featureBuffer = if (featureFile.startsWith("hdfs://")) {
      ByteBuffer.wrap(readHdfsByte(featureFile))
    } else {
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(featureFile)))
    }
    val labelBuffer = if (featureFile.startsWith("hdfs://")) {
      ByteBuffer.wrap(readHdfsByte(labelFile))
    } else {
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(labelFile)))
    }
    val labelMagicNumber = labelBuffer.getInt()

    require(labelMagicNumber == 2049)
    val featureMagicNumber = featureBuffer.getInt()
    require(featureMagicNumber == 2051)

    val labelCount = labelBuffer.getInt()
    val featureCount = featureBuffer.getInt()
    require(labelCount == featureCount)

    val rowNum = featureBuffer.getInt()
    val colNum = featureBuffer.getInt()

    val images = new Array[Array[Double]](featureCount / 600)
    val label = new Array[Double](labelCount / 600)
    var i = 0
    while (i < featureCount / 600) {
      val img = new Array[Double](rowNum * colNum)
      var y = 0
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          img(x + y * colNum) = (featureBuffer.get() & 0xff) / 255.0f
          x += 1
        }
        y += 1
      }
      images(i) = img
      label(i) = labelBuffer.get().toDouble
      i += 1
    }

    (images, label)
  }
}

