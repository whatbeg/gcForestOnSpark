/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package org.apache.spark.ml.classification

import java.text.SimpleDateFormat
import java.util.Date

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.Helper.{Accuracy, UserDefinedFunctions => UDF}
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.tree.GCForestParams
import org.apache.spark.ml.util._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, LongType, StructField, StructType}

import scala.collection.mutable.ArrayBuffer


class GCForestClassifier(override val uid: String)
  extends ProbabilisticClassifier[Vector, GCForestClassifier, GCForestClassificationModel]
  with DefaultParamsWritable with GCForestParams {

  def this() = this(Identifiable.randomUID("gcf"))

  /**
    * Scan Sequence Data
    * @param dataset raw input label and features
    * @param windowSize the window size
    * @return
    */
  def extractSequenceRDD(dataset: Dataset[_], windowSize: Int): DataFrame = {
    require(getDataSize.length == 1, "You must set Sequence length by setDataSize")
    val sparkSession = dataset.sparkSession
    val schema = dataset.schema

    val seqSize = getDataSize(0)
    require(seqSize >= windowSize, "Window Size must be smaller than Sequence Length")

    val vector2Dense = udf { (features: Vector, w: Int) =>
      new DenseVector(features.toArray)
    }
    val windowInstances = dataset.withColumn($(featuresCol), vector2Dense(col($(featuresCol))))
      .rdd.flatMap { row =>
        val rows = Array.fill[Row](seqSize - windowSize + 1)(null)
        val featureIdx = row.fieldIndex($(featuresCol))
        val vector = row.getAs[Vector]($(featuresCol))
        val features = Array.fill[Double](windowSize)(0d)
        val newRow = ArrayBuffer[Any]() ++= row.toSeq
        val windowNumIdx = newRow.length
        newRow += 0

        Range(0, seqSize - windowSize + 1).foreach { offset =>
          Range(0, windowSize).foreach { x =>
            features(x) = vector(x + offset)
          }
          val windowNum = offset
          newRow(featureIdx) = new DenseVector(features)
          newRow(windowNumIdx) = windowNum.toLong
          rows(windowNum) = Row.fromSeq(newRow)
        }
        rows
    }

    val newSchema = schema
      .add(StructField($(windowCol), LongType))
    sparkSession.createDataFrame(windowInstances, newSchema)

  }
  /**
    * Scan image-style data
    * @param dataset raw input features
    * @param windowWidth the width of window
    * @param windowHeight the height of window
    * @return
    */
  def extractMatrixRDD(dataset: Dataset[_], windowWidth: Int, windowHeight: Int): DataFrame = {
    require(getDataSize.length == 2, "You must set Image resolution by setDataSize")
    val sparkSession = dataset.sparkSession
    val schema = dataset.schema

    val width = getDataSize(0)  // 10
    val height = getDataSize(1)  // 10

    require(width >= windowWidth, "The width of image must be greater than window's")
    require(height >= windowHeight, "The height of image must be greater than window's")

    val vector2Matrix = udf { (features: Vector, w: Int, h: Int) =>
      new DenseMatrix(w, h, features.toArray)
    }

    val windowInstances = dataset.withColumn($(featuresCol),
      vector2Matrix(col($(featuresCol)), lit(width), lit(height)))
      .rdd.flatMap { row =>
          val rows = Array.fill[Row]((width - windowWidth + 1) * (height - windowHeight + 1))(null)
          val featureIdx = row.fieldIndex($(featuresCol))
          val matrix = row.getAs[DenseMatrix]($(featuresCol))
          val features = Array.fill[Double](windowWidth * windowHeight)(0d)
          val newRow = ArrayBuffer[Any]() ++= row.toSeq
          val windowNumIdx = newRow.length
          newRow += 0

          Range(0, width - windowWidth + 1).foreach { x_offset =>
            Range(0, height - windowHeight + 1).foreach { y_offset =>
              Range(0, windowWidth).foreach { x =>
                Range(0, windowHeight).foreach { y =>
                  features(x * windowWidth + y) = matrix(x + x_offset, y + y_offset)
                }
              }
              val windowNum = x_offset * (width - windowWidth + 1) + y_offset
              newRow(featureIdx) = new DenseVector(features)
              newRow(windowNumIdx) = windowNum.toLong
              rows(windowNum) = Row.fromSeq(newRow)
            }
          }
          rows
    }

    val newSchema = schema
      .add(StructField($(windowCol), LongType))

    sparkSession.createDataFrame(windowInstances, newSchema)
  }

  /**
    * Use cross-validation to build k classes distribution features
    * @param training training dataset
    * @param testing testing dataset
    * @param rfc random forest classifier
    * @return k classes distribution features and random forest model
    */
  def featureTransform(training: Dataset[_], testing: Dataset[_], rfc: RandomForestClassifier):
  (DataFrame, DataFrame, RandomForestCARTModel) = {
    val schema = training.schema
    val sparkSession = training.sparkSession
    var out_train: DataFrame = null
    var out_test: DataFrame = null

    if (testing != null) require(training.schema.equals(testing.schema))
    val testingDataset = if (testing == null) null else testing.toDF

    // cross-validation for k classes distribution features
    val splits = MLUtils.kFold(training.toDF.rdd, $(numFolds), $(seed))
    splits.zipWithIndex.foreach {
      case ((t, v), splitIndex) =>
        val trainingDataset = sparkSession.createDataFrame(t, schema).cache
        val validationDataset = sparkSession.createDataFrame(v, schema).cache
        val model = rfc.fit(trainingDataset)

        trainingDataset.unpersist()
        // rawPrediction == probabilityCol
        val val_result = model.transform(validationDataset)
          .drop($(featuresCol)).drop($(rawPredictionCol)).drop($(predictionCol))
          .withColumnRenamed($(probabilityCol), $(featuresCol))
        out_train = if (out_train == null) val_result else out_train.union(val_result)
        if (testing != null) {
          val test_result = model.transform(testingDataset)
            .drop($(featuresCol)).drop($(rawPredictionCol)).drop($(predictionCol))
            .withColumnRenamed($(probabilityCol), $(featuresCol)+s"$splitIndex")
          out_test = if (out_test == null) test_result
                     else out_test.join(test_result, Seq($(instanceCol), $(labelCol)))
        }
        validationDataset.unpersist()
    }
    if (testing != null) testingDataset.unpersist()
    out_test = out_test.withColumn($(featuresCol),
      UDF.mergeVectorForKfold(3)(Range(0, $(numFolds)).map(k => col($(featuresCol) + s"$k")): _*))
      .select($(instanceCol), $(labelCol), $(featuresCol))

    (out_train, out_test, rfc.fit(training))
  }

  // create a random forest classifier by type
  def genRFClassifier(rfType: String,
                      treeNum: Int,
                      minInstancePerNode: Int): RandomForestClassifier = {
    val rf = rfType match {
      case "rfc" => new RandomForestClassifier()
      case "crfc" => new CompletelyRandomForestClassifier()
    }

    rf.setNumTrees(treeNum)
      .setMaxBins($(randomForestMaxBins))
      .setMaxDepth($(randomForestMaxDepth))
      .setMinInstancesPerNode(minInstancePerNode)
      .setFeatureSubsetStrategy("sqrt")
  }

  /**
    * Concat multi-scan features
    * @param dataset one of a window
    * @param sets the others
    * @return input for Cascade Forest
    */
  def concatenate(dataset: Dataset[_], sets: Dataset[_]*): DataFrame = {
    val sparkSession = dataset.sparkSession
    var unionSet = dataset.toDF
    sets.foreach(ds => unionSet = unionSet.union(ds.toDF))

    class Record(val instance: Long,    // instance id
                 val label: Double,     // label
                 val features: Vector,  // features
                 val scanId: Int,       // the scan id for multi-scan
                 val treeNum: Int,      // tree id
                 val winId: Long) extends Serializable // window id

    val concatData = unionSet.select(
      $(instanceCol), $(labelCol),
      $(featuresCol), $(scanCol),
      $(forestNumCol), $(windowCol)).rdd.map {
      row =>
        val instance = row.getAs[Long]($(instanceCol))
        val label = row.getAs[Double]($(labelCol))
        val features = row.getAs[Vector]($(featuresCol))
        val scanId = row.getAs[Int]($(scanCol))
        val treeNum = row.getAs[Int]($(forestNumCol))
        val winId = row.getAs[Long]($(windowCol))

        new Record(instance, label, features, scanId, treeNum, winId)
    }.groupBy(
      record => record.instance
    ).map { group =>
      val instance = group._1
      val records = group._2
      val label = records.head.label

      def recordCompare(left: Record, right: Record): Boolean = {
        var code = left.scanId.compareTo(right.scanId)
        if (code == 0) code = left.treeNum.compareTo(right.treeNum)
        if (code == 0) code = left.winId.compareTo(right.winId)
        code < 0
      }

      val features = new DenseVector(records.toSeq.sortWith(recordCompare)
        .flatMap(_.features.toArray).toArray)
      // features = [0, 0, ..., 0] (903 dim)
      Row.fromSeq(Array[Any](instance, label, features))
    }

    val schema: StructType = StructType(Seq[StructField]())
      .add(StructField($(instanceCol), LongType))
      .add(StructField($(labelCol), DoubleType))
      .add(StructField($(featuresCol), new VectorUDT))
    sparkSession.createDataFrame(concatData, schema)
  }

  /**
    * concat inputs of Cascade Forest with prediction
    * @param feature input features
    * @param predict prediction features
    * @return
    */
  def mergeFeatureAndPredict(feature: Dataset[_], predict: Dataset[_]): DataFrame = {
    val vectorMerge = udf { (v1: Vector, v2: Vector) =>
      new DenseVector(v1.toArray.union(v2.toArray))
    }

    if (predict != null) {
      feature.join(
        // join (predict feature col to predictionCol)
        predict.withColumnRenamed($(featuresCol), $(predictionCol)),
        Seq($(instanceCol))  // join on instanceCol
        // replace featureCol with featureCol + predictionCol
      ).withColumn($(featuresCol), vectorMerge(col($(featuresCol)), col($(predictionCol)))
      ).select($(instanceCol), $(featuresCol), $(labelCol)).toDF  // select 3 cols to DataFrame
    } else {
      feature.toDF
    }
  }

  def getNowTime(): String = {
    val now: Date = new Date()
    val dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss,SS")
    val hehe = dateFormat.format( now )
    hehe
  }

//  def getAccuracy(phase: String, layer: Int, lastPrediction: DataFrame, numClasses: Int): Unit = {
//    val rightCount = lastPrediction.rdd.map { row =>
//      val avgPredict = Array.fill[Double](numClasses)(0d)
//      val label = row.getAs[Double]($(labelCol))
//      val features = row.getAs[Vector](${featuresCol}).toArray
//      features.indices.foreach{ i =>
//        val classType = i % numClasses
//        avgPredict(classType) = avgPredict(classType) + features(i)
//      }
//      val predict = new DenseVector(avgPredict).argmax
////      println(predict.toInt + " x " + label.toInt)
//      if (predict.toInt == label.toInt) 1 else 0
//    }.reduce((l, r) => l+r)
//
//    val totalCount = lastPrediction.count()
//    val accuracy = rightCount / (if (totalCount != 0) totalCount.toDouble else 1.0)
//    println("[%s] Cascade Layer [%d] Accuracy: (%d / %d = %.3f%%)".format(
//      getNowTime(), layer, rightCount, totalCount, accuracy))
//    logInfo("[%s] Cascade Layer [%d] Accuracy: (%d / %d = %.3f%%)".format(
//      getNowTime(), layer, rightCount, totalCount, accuracy))
//  }

  def multi_grain_Scan(dataset: Dataset[_]): (DataFrame, Array[MultiGrainedScanModel]) = {
    require(dataset != null, "Null dataset need not to scan")
    var scanFeature: DataFrame = null
    val mgsModels = ArrayBuffer[MultiGrainedScanModel]()

    /**
      *  Multi-Grained Scanning
      */
    println(s"[${getNowTime()}] Multi Grained Scanning begin!")
    if ($(dataStyle) == "image" && $(multiScanWindow).length > 0) {
      require($(multiScanWindow).length % 2 == 0,
        "The multiScanWindow must has the even number for image-style data")

      val scanFeatures = ArrayBuffer[Dataset[_]]()

      Range(0, $(multiScanWindow).length / 2).foreach { i =>
        // Get the size of scan window
        val (w, h) = ($(multiScanWindow)(i), $(multiScanWindow)(i+1))
        val windowInstances = extractMatrixRDD(dataset, w, h)
        val rfc = genRFClassifier("rfc", $(scanForestTreeNum), $(scanForestMinInstancesPerNode))
        var (rfcFeature, _, rfcModel) = featureTransform(windowInstances, null, rfc)
        rfcFeature = rfcFeature.withColumn($(forestNumCol), lit(1)).withColumn($(scanCol), lit(i))
        scanFeatures += rfcFeature

        val crfc =
          genRFClassifier("crfc", $(scanForestTreeNum), $(scanForestMinInstancesPerNode))
        var (crfcFeature, _, crfcModel) = featureTransform(windowInstances, null, crfc)
        crfcFeature = crfcFeature.withColumn($(forestNumCol), lit(2)).withColumn($(scanCol), lit(i))
        scanFeatures += crfcFeature

        mgsModels += new MultiGrainedScanModel(Array(w, h), rfcModel, crfcModel)
      }
      scanFeature = concatenate(scanFeatures.head, scanFeatures.tail:_*).cache
    } else if ($(dataStyle) == "sequence" && $(multiScanWindow).length > 0) { // TODO
      val scanFeatures = ArrayBuffer[Dataset[_]]()
      $(multiScanWindow).indices.foreach { i => // for each window
        val windowSize = $(multiScanWindow)(i)
        val windowInstances = extractSequenceRDD(dataset, windowSize)
        val rfc = genRFClassifier("rfc", $(scanForestTreeNum), $(scanForestMinInstancesPerNode))
        var (rfcFeature, _, rfcModel) = featureTransform(windowInstances, null, rfc)
        rfcFeature = rfcFeature.withColumn($(forestNumCol), lit(1)).withColumn($(scanCol), lit(i))
        scanFeatures += rfcFeature

        val crfc =
          genRFClassifier("crfc", $(scanForestTreeNum), $(scanForestMinInstancesPerNode))
        var (crfcFeature, _, crfcModel) = featureTransform(windowInstances, null, crfc)
        crfcFeature = crfcFeature.withColumn($(forestNumCol), lit(2)).withColumn($(scanCol), lit(i))
        scanFeatures += crfcFeature

        mgsModels += new MultiGrainedScanModel(Array(windowSize), rfcModel, crfcModel)
      }
      scanFeature = concatenate(scanFeatures.head, scanFeatures.tail:_*).cache
    } else if ($(multiScanWindow).length > 0){
      throw new UnsupportedOperationException(
        "The dataStyle : " + $(dataStyle) + " is unsupported!")
    }

    if ($(multiScanWindow).length == 0)
      scanFeature = dataset.toDF
    // scanFeature: (instanceId, label, feature)
    println(s"[${getNowTime()}] Multi Grained Scanning finished!")
    (scanFeature, mgsModels.toArray)
  }

  override protected def train(dataset: Dataset[_]): GCForestClassificationModel = {
    train(dataset, null)
  }

  private def train(dataset: Dataset[_], testset: Dataset[_]): GCForestClassificationModel = {
    val numClasses: Int = getNumClasses(dataset)
    val erfModels = ArrayBuffer[Array[RandomForestCARTModel]]()

    val (scanFeature_train, mgsModels) = multi_grain_Scan(dataset)

    val (scanFeature_test, mgsModels_test) =
      if (testset != null) multi_grain_Scan(testset) else (null, null)

    /**
      *  Cascade Forest
     */
    println(s"[${getNowTime()}] Cascade Forest begin!")
    val sparkSession = scanFeature_train.sparkSession
    var lastPrediction: DataFrame = null
    var lastPrediction_test: DataFrame = null
    val acc_list = Array(ArrayBuffer[Double](), ArrayBuffer[Double]())

    // Init classifiers
    val maxIteration = $(cascadeForestMaxIteration)
    require(maxIteration > 0, "Zero maxIteration")
    var layer_id = 1
    var reachMaxLayer = false
    while (!reachMaxLayer) {
      println(s"[${getNowTime()}] Cascade Forest Layer ${layer_id}")
      val ensembleRandomForest = Array[RandomForestClassifier](
        genRFClassifier("rfc", $(cascadeForestTreeNum), $(cascadeForestMinInstancesPerNode)),
        genRFClassifier("rfc", $(cascadeForestTreeNum), $(cascadeForestMinInstancesPerNode)),
        genRFClassifier("rfc", $(cascadeForestTreeNum), $(cascadeForestMinInstancesPerNode)),
        genRFClassifier("rfc", $(cascadeForestTreeNum), $(cascadeForestMinInstancesPerNode)),
        genRFClassifier("crfc", $(cascadeForestTreeNum), $(cascadeForestMinInstancesPerNode)),
        genRFClassifier("crfc", $(cascadeForestTreeNum), $(cascadeForestMinInstancesPerNode)),
        genRFClassifier("crfc", $(cascadeForestTreeNum), $(cascadeForestMinInstancesPerNode)),
        genRFClassifier("crfc", $(cascadeForestTreeNum), $(cascadeForestMinInstancesPerNode))
      )

      val training = mergeFeatureAndPredict(scanFeature_train, lastPrediction)
      val testing = mergeFeatureAndPredict(scanFeature_test, lastPrediction_test)
      val n_train = training.count()
      val n_test = testing.count()
      val features_dim = training.first().mkString.split(",").length
      // val fts = training.collectAsList().get(0)
      println(s"[${getNowTime()}] Training Set = ($n_train, $features_dim), " +
        s"Testing Set = ($n_test, $features_dim)")

      var ensemblePredict: DataFrame = null
      var ensemblePredict_test: DataFrame = null
      println(s"[${getNowTime()}] Forests fitting and transforming ......")
      erfModels += ensembleRandomForest.indices.map { it =>
        val transformed = featureTransform(training, testing, ensembleRandomForest(it))
        val predict = transformed._1
          .withColumn($(forestNumCol), lit(it))
          .select($(instanceCol), $(labelCol), $(featuresCol), $(forestNumCol))
        val predict_test = transformed._2
          .withColumn($(forestNumCol), lit(it))
          .select($(instanceCol), $(labelCol), $(featuresCol), $(forestNumCol))
        ensemblePredict =
        if (ensemblePredict == null) predict else ensemblePredict.toDF.union(predict)
        ensemblePredict_test =
        if (ensemblePredict_test == null) predict_test else ensemblePredict_test
          .toDF.union(predict_test)
        predict.unpersist()
        predict_test.unpersist()
        transformed._3
      }.toArray
      println(s"[${getNowTime()}] Forests fitting and transforming finished!")

      val schema = new StructType()
        .add(StructField($(instanceCol), LongType))
        .add(StructField($(featuresCol), new VectorUDT))

      println(s"[${getNowTime()}] Getting prediction RDD ......")
      val predictRDDs =
        Array(ensemblePredict, ensemblePredict_test).zipWithIndex.map { case (predict, idx) =>
        val grouped = predict.rdd.groupBy(_.getAs[Long]($(instanceCol)))
        val predictRDD = grouped.map { group =>
          val instanceId = group._1
          val rows = group._2
          val features = new DenseVector(rows.toArray
            .sortWith(_.getAs[Int]($(forestNumCol)) < _.getAs[Int]($(forestNumCol)))
            .flatMap(_.getAs[Vector]($(featuresCol)).toArray))
          Row.fromSeq(Array[Any](instanceId, features))
        }
        val rightCount = grouped.map { group =>
          val rows = group._2
          val label = rows.head.getAs[Double]($(labelCol))
          val features = new DenseVector(rows.toArray  // do not need to be put in order
            .flatMap(_.getAs[Vector]($(featuresCol)).toArray)).toArray
          val avgPredict = Array.fill[Double](numClasses)(0d)
          features.indices.foreach{ i =>
            val classType = i % numClasses
            avgPredict(classType) = avgPredict(classType) + features(i)
          }
          val predict_result = avgPredict.zipWithIndex.maxBy(_._1)._2
          if (label.toInt == predict_result) 1 else 0
        }.reduce((l, r) => l+r)
        val totalCount = if (idx == 0) n_train else n_test
        val accuracy = new Accuracy(rightCount, totalCount)
        println(s"\n[${getNowTime()}] Cascade Layer [${layer_id}]" +
          s" ${if (idx == 0) "Training" else "Testing"} Set Average ${accuracy.toString}")
        acc_list(idx) += accuracy.getAccuracy
        predictRDD
      }
      // =======================================================================
//      val grouped = ensemblePredict.rdd.groupBy(_.getAs[Long]($(instanceCol)))
//      val predictionRDD = grouped.map { group =>
//        val instanceId = group._1
//        val rows = group._2
//        val features = new DenseVector(rows.toArray
//          .sortWith(_.getAs[Int]($(treeNumCol)) < _.getAs[Int]($(treeNumCol)))
//          .flatMap(_.getAs[Vector]($(featuresCol)).toArray))
//        Row.fromSeq(Array[Any](instanceId, features))
//      }
//      println(s"[${getNowTime()}] Get prediction RDD finished!")
//      val rightCount = grouped.map { group =>
//        val rows = group._2
//        val label = rows.head.getAs[Double]($(labelCol))
//        val features = new DenseVector(rows.toArray
//          .sortWith(_.getAs[Int]($(treeNumCol)) < _.getAs[Int]($(treeNumCol)))
//          .flatMap(_.getAs[Vector]($(featuresCol)).toArray)).toArray
//        val avgPredict = Array.fill[Double](numClasses)(0d)
//        features.indices.foreach{ i =>
//          val classType = i % numClasses
//          avgPredict(classType) = avgPredict(classType) + features(i)
//        }
//        val predict = new DenseVector(avgPredict).argmax
//        if (label.toInt == predict) 1 else 0
//      }.reduce((l, r) => l+r)
//      val totalCount = predictionRDD.count()
//      val accuracy = new Accuracy(rightCount, totalCount)
//      println("\n[%s] Cascade Layer [%d] %s\n".format(
//        getNowTime(), layer_id, accuracy.toString))
//      logInfo("[%s] Cascade Layer [%d] %s".format(
//        getNowTime(), layer_id, accuracy.toString))
//      train_acc_list += accuracy.getAccuracy
      // ==============================================================================
      val opt_layer_id_train = acc_list(0).zipWithIndex.maxBy(_._1)._2
      val opt_layer_id_test = acc_list(1).zipWithIndex.maxBy(_._1)._2
      lastPrediction = sparkSession.createDataFrame(predictRDDs(0), schema)
      lastPrediction_test = sparkSession.createDataFrame(predictRDDs(1), schema)
      reachMaxLayer =
        (layer_id == maxIteration) ||
          ($(earlyStopByTest) && layer_id - opt_layer_id_test >= $(earlyStoppingRounds)) ||
          (!$(earlyStopByTest) && layer_id - opt_layer_id_train >= $(earlyStoppingRounds))
      layer_id += 1
    }

    scanFeature_train.unpersist
    if (testset != null) scanFeature_test.unpersist
    // logger.info("Cascade Forest finished!")
    new GCForestClassificationModel(mgsModels, erfModels.toArray, numClasses)
  }

  private def get_optimal_layer_id(doubles: ArrayBuffer[Double]): Int = {
    var ma = -1.0
    var idx = 0
    for ((e, i) <- doubles.zipWithIndex) {
      ma = if (e > ma) e else ma
      idx = if (e > ma) i else idx
    }
    idx + 1
  }

  override def fit(dataset: Dataset[_]): GCForestClassificationModel = super.fit(dataset)

  def fit(trainset: Dataset[_], testset: Dataset[_]): GCForestClassificationModel = {
    // This handles a few items such as schema validation.
    // Developers only need to implement train().
    transformSchema(trainset.schema, logging = true)
    transformSchema(testset.schema, logging = true)

    // Cast LabelCol to DoubleType and keep the metadata.
    val labelMeta = trainset.schema($(labelCol)).metadata
    val casted_train =
      trainset.withColumn($(labelCol), col($(labelCol)).cast(DoubleType), labelMeta)

    val labelMeta_test = testset.schema($(labelCol)).metadata
    val casted_test =
      testset.withColumn($(labelCol), col($(labelCol)).cast(DoubleType), labelMeta_test)

    copyValues(train(casted_train, casted_test))
  }

  override def copy(extra: ParamMap): GCForestClassifier = defaultCopy(extra)
}


class GCForestClassificationModel private[ml] (
   override val uid: String,
   private val scanModel: Array[MultiGrainedScanModel],
   private val cascadeForest: Array[Array[RandomForestCARTModel]],
   override val numClasses: Int)
  extends ProbabilisticClassificationModel[Vector, GCForestClassificationModel]
    with GCForestParams with MLWritable with Serializable {

  def this(scanModel: Array[MultiGrainedScanModel],
           cascadeForest: Array[Array[RandomForestCARTModel]],
           numClasses: Int) =
    this(Identifiable.randomUID("gcfc"), scanModel, cascadeForest, numClasses)

  val numScans: Int = scanModel.length
  val numCascades: Int = cascadeForest.length

  def predictScanFeature(features: Vector): Vector = {
    val avgPredict = Array.fill[Double](numClasses)(0d)
    var lastPredict = Array[Double]()

    cascadeForest.foreach { models =>
      lastPredict = models.flatMap(
        m => m.predictProbability(new DenseVector(features.toArray.union(lastPredict))).toArray
      )
    }

    lastPredict.indices.foreach { i =>
      val classType = i % numClasses
      avgPredict(classType) = avgPredict(classType) + lastPredict(i)
    }

    new DenseVector(avgPredict)
  }

  override def predictRaw(features: Vector): Vector = {
    val scanFeatures = ArrayBuffer[Double]()

    /**
      * Multi-Grained Scanning
      */
    if ($(dataStyle) == "image") {
      val width = $(dataSize)(0)
      val height = $(dataSize)(1)
      require(features.size == width * height)

      val matrix = new DenseMatrix(width, height, features.toArray)

      scanModel.foreach { model =>
        val windowWidth = model.windows(0)
        val windowHeight = model.windows(1)
        val windowFeatures = Array.fill[Double](windowWidth * windowHeight)(0)

        Seq(model.rfcModel, model.crfcModel).foreach { featureModel =>
          Range(0, width - windowWidth + 1).foreach { x_offset =>
            Range(0, height - windowHeight + 1).foreach { y_offset =>
              Range(0, windowWidth).foreach { x =>
                Range(0, windowHeight).foreach { y =>
                  windowFeatures(x * windowWidth + y) = matrix(x + x_offset, y + y_offset)
                }
              }
              scanFeatures ++=
                featureModel.predictProbability(new DenseVector(windowFeatures)).toArray
            }
          }
        }
      }
    } else if ($(dataStyle) == "sequence") { // TODO
      throw new UnsupportedOperationException("Unsupported sequence data rightly!")
    } else {
      throw new UnsupportedOperationException(
        "The dataStyle : " + $(dataStyle) + " is unsupported!")
    }

    /**
      *  Cascade Predicting
      */
    predictScanFeature(new DenseVector(scanFeatures.toArray))
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        ProbabilisticClassificationModel.normalizeToProbabilitiesInPlace(dv)
        dv
      case _: SparseVector =>
        throw new RuntimeException("Unexpected error in GCForestClassificationModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  override protected def transformImpl(dataset: Dataset[_]): DataFrame = {
    val bcastModel = dataset.sparkSession.sparkContext.broadcast(this)
    val predictUDF = udf { (features: Any) =>
      bcastModel.value.predict(features.asInstanceOf[Vector])
    }
    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }

  override def copy(extra: ParamMap): GCForestClassificationModel = {
    copyValues(new GCForestClassificationModel(uid,  scanModel, cascadeForest, numClasses), extra)
  }

  override def write: MLWriter =
    new GCForestClassificationModel.GCForestClassificationModelWriter(this)
}

/**
  * The metadata of GCForestClassificationModel
  *
  * root                  // the root directory of GCForestClassificationModel
  *  |--metadata          // metadata of GCForestClassificationModel
  *  |--scan              // Multi-Grained Scanning
  *  |   |--0
  *  |   |  |--metadata
  *  |   |  |--rfc
  *  |   |  |--crfc
  *  |   |--1
  *  |   |--2
  *  |--cascade           // Cascade Forest
  *  |   |--0             // the level of Cascade Forest
  *  |   |  |--0          // the number of Forest
  *  |   |  |--1
  *  |   |  |--2
  *  |   |  |--3
  *  |   |--1
  *  |   |--2
  */
object GCForestClassificationModel extends MLReadable[GCForestClassificationModel] {
  override def read: MLReader[GCForestClassificationModel] = new GCForestClassificationModelReader

  override def load(path: String): GCForestClassificationModel = super.load(path)

  private[GCForestClassificationModel]
  class GCForestClassificationModelWriter(instance: GCForestClassificationModel)
    extends MLWriter {

    override protected def saveImpl(path: String): Unit = {

      val gcMetadata: JObject = Map(
        "numClasses" -> instance.numClasses,
        "numScans" -> instance.numScans,
        "numCascades" -> instance.numCascades)
      DefaultParamsWriter.saveMetadata(instance, path, sparkSession.sparkContext, Some(gcMetadata))

      // scanModel
      val scanPath = new Path(path, "scan").toString
      instance.scanModel.zipWithIndex.foreach { case (model, index) =>
        val modelPath = new Path(scanPath, index.toString).toString
        val metadata: JObject = Map(
          "windows" -> model.windows.toList
        )
        DefaultParamsWriter
          .saveMetadata(model, modelPath, sparkSession.sparkContext, Some(metadata))
        val rfModelPath = new Path(modelPath, "rf").toString
        model.rfcModel.save(rfModelPath)
        val crtfModelPath = new Path(modelPath, "crtf").toString
        model.crfcModel.save(crtfModelPath)
      }

      // CascadeForestModel
      val cascadePath = new Path(path, "cascade").toString
      instance.cascadeForest.zipWithIndex.foreach { case(models, level) =>
        val modelsPath = new Path(cascadePath, level.toString).toString
        models.zipWithIndex.foreach { case(model, index) =>
          val modelPath = new Path(modelsPath, index.toString).toString
          model.save(modelPath)
        }
      }
    }
  }

  private class GCForestClassificationModelReader
    extends MLReader[GCForestClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[GCForestClassificationModel].getName
    val mgsClassName: String = classOf[MultiGrainedScanModel].getName

    override def load(path: String): GCForestClassificationModel = {
      implicit val format = DefaultFormats
      val gcMetadata = DefaultParamsReader.loadMetadata(path, sparkSession.sparkContext, className)

      val numClasses = (gcMetadata.metadata \ "numClasses").extract[Int]
      val numScans = (gcMetadata.metadata \ "numScans").extract[Int]
      val numCascades = (gcMetadata.metadata \ "numCascades").extract[Int]

      val scanPath = new Path(path, "scan").toString
      val scanModel = Range(0, numScans).map { index =>
        val modelPath = new Path(scanPath, index.toString).toString
        val scanMetadata = DefaultParamsReader
          .loadMetadata(path, sparkSession.sparkContext, mgsClassName)
        val windows = (scanMetadata.metadata \ "windows").extract[Array[Int]]
        val rfPath = new Path(modelPath, "rf").toString
        val rfModel = RandomForestCARTModel.load(rfPath)
        val crtfPath = new Path(modelPath, "crtf").toString
        val crtfModel = RandomForestCARTModel.load(crtfPath)
        new MultiGrainedScanModel(windows, rfModel, crtfModel)
      }.toArray

      val cascadePath = new Path(path, "cascade").toString
      val cascadeForest = Range(0, numCascades).map { level =>
        val modelsPath = new Path(cascadePath, level.toString).toString
        Range(0, 4).map { index =>
          val modelPath = new Path(modelsPath, index.toString).toString
          RandomForestCARTModel.load(modelPath)
        }.toArray
      }.toArray

      val gcForestModel =
        new GCForestClassificationModel(gcMetadata.uid, scanModel, cascadeForest, numClasses)

      DefaultParamsReader.getAndSetParams(gcForestModel, gcMetadata)
      gcForestModel
    }
  }
}

class MultiGrainedScanModel(override val uid: String,
                            val windows: Array[Int],
                            val rfcModel: RandomForestCARTModel,
                            val crfcModel: RandomForestCARTModel) extends Params {
  def this(windows: Array[Int],
           rfcModel: RandomForestCARTModel,
           crfcModel: RandomForestCARTModel) =
    this(Identifiable.randomUID("mgs"), windows, rfcModel, crfcModel)

  override def copy(extra: ParamMap): Params =
    copyValues(new MultiGrainedScanModel(uid, windows, rfcModel, crfcModel), extra)
}