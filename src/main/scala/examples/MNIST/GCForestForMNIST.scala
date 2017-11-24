/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package examples.MNIST

import org.apache.spark.ml.classification.GCForestClassifier
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, DoubleType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}


object GCForestForMNIST {
  def main(args: Array[String]): Unit = {

    import Utils._
    val folder = "data/mnist"
    val trainData = folder + "/train-images-idx3-ubyte"
    val trainLabel = folder + "/train-labels-idx1-ubyte"
    val validationData = folder + "/t10k-images-idx3-ubyte"
    val validationLabel = folder + "/t10k-labels-idx1-ubyte"

    val output = "data/mnist_model"

    val spark = SparkSession.builder().appName(this.getClass.getSimpleName).master("local[*]").getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    val (train_img, train_label) = load(trainData, trainLabel)

    val trainRDD = spark.sparkContext.parallelize(train_img.indices).map { idx =>
      val label = train_label(idx)
      val features = train_img(idx)
      Row.fromSeq(Seq[Any](label, features))
    }

    val schema = new StructType()
      .add(StructField("label", DoubleType))
      .add(StructField("features", ArrayType(DoubleType)))

    val arr2vec = udf {(features: Seq[Double]) => new DenseVector(features.toArray)}
    val train = spark.createDataFrame(trainRDD, schema)
      .withColumn("features", arr2vec(col("features")))

    val gcForest = new GCForestClassifier()
      .setDataSize(Array(28, 28))
      .setDataStyle("image")
      .setMultiScanWindow(Array(7, 7))
      .setCascadeForestTreeNum(5)
      .setScanForestTreeNum(5)
      .setMaxIteration(1)

    // val commonForest = new CompleteRandomTreeForestClassifier()
    val model = gcForest.fit(train)
    // val model = commonForest.fit(train)
    // model.save(output)
  }
}