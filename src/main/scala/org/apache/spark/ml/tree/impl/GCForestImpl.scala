/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.tree.impl

import java.text.SimpleDateFormat
import java.util.Date

import org.apache.spark.HashPartitioner
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.{Accuracy, Evaluator, Metric}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Vector, VectorUDT}
import org.apache.spark.ml.tree.configuration.GCForestStrategy
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.Helper.{UserDefinedFunctions => UDF}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types.{DoubleType, LongType, StructField, StructType}
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

private[spark] object GCForestImpl extends Logging {

  def run(
      input: Dataset[_],
      gcforestStategy: GCForestStrategy
      ): GCForestClassificationModel = {
    train(input, strategy = gcforestStategy)
  }

  def runWithValidation(
      input: Dataset[_],
      validationInput: Dataset[_],
      gCForestStrategy: GCForestStrategy
      ): GCForestClassificationModel = {
    trainWithValidation(input, validationInput, gCForestStrategy)
  }

  val dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss,SSS")

  /**
    * Scan Sequence Data
    * @param dataset raw input label and features
    * @param windowSize the window size
    * @return
    */
  def extractSequenceRDD(dataset: Dataset[_],
                         windowSize: Int,
                         dataSize: Array[Int],
                         featuresCol: String,
                         winCol: String): DataFrame = {
    require(dataSize.length == 1, "You must set Sequence length by setDataSize")
    val sparkSession = dataset.sparkSession
    val schema = dataset.schema

    val seqSize = dataSize(0)
    require(seqSize >= windowSize, "Window Size must be smaller than Sequence Length")

    val vector2Dense = udf { (features: Vector) =>
      features.asInstanceOf[DenseVector]
    }

    val windowInstances = dataset.withColumn(featuresCol, vector2Dense(col(featuresCol)))
      .rdd.flatMap { row =>
      val rows = Array.fill[Row](seqSize - windowSize + 1)(null)
      val featureIdx = row.fieldIndex(featuresCol)
      val vector = row.getAs[Vector](featuresCol)
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
      .add(StructField(winCol, LongType))
    sparkSession.createDataFrame(windowInstances, newSchema)

  }

  /**
    * Scan image-style data
    * @param dataset raw input features
    * @param windowWidth the width of window
    * @param windowHeight the height of window
    * @return
    */
  def extractMatrixRDD(dataset: Dataset[_],
                       windowWidth: Int,
                       windowHeight: Int,
                       dataSize: Array[Int],
                       featuresCol: String,
                       winCol: String): DataFrame = {
    require(dataSize.length == 2, "You must set Image resolution by setDataSize")
    val sparkSession = dataset.sparkSession
    val schema = dataset.schema

    val width = dataSize(0)  // 10
    val height = dataSize(1)  // 10

    require(width >= windowWidth, "The width of image must be greater than window's")
    require(height >= windowHeight, "The height of image must be greater than window's")

    val vector2Matrix = udf { (features: Vector, w: Int, h: Int) =>
      new DenseMatrix(w, h, features.toArray)
    }

    val windowInstances = dataset.withColumn(featuresCol,
      vector2Matrix(col(featuresCol), lit(width), lit(height)))
      .rdd.flatMap { row =>
      val rows = Array.fill[Row]((width - windowWidth + 1) * (height - windowHeight + 1))(null)
      val featureIdx = row.fieldIndex(featuresCol)
      val matrix = row.getAs[DenseMatrix](featuresCol)
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
      .add(StructField(winCol, LongType))

    sparkSession.createDataFrame(windowInstances, newSchema)
  }

  def cvClassVectorGenerator(
      training: Dataset[_],
      rfc: RandomForestCARTClassifier,
      numFolds: Int,
      seed: Long,
      strategy: GCForestStrategy,
      isScan: Boolean = false,
      message: String = ""):
  (DataFrame, Metric, RandomForestCARTModel) = {
    val schema = training.schema
    val sparkSession = training.sparkSession
    var out_train: DataFrame = null // closure need

    // cross-validation for k classes distribution features
    var train_metric = new Accuracy(0, 0)
    val splits = MLUtils.kFold(training.toDF.rdd, numFolds, seed * System.currentTimeMillis())
    splits.zipWithIndex.foreach {
      case ((t, v), splitIndex) =>
        val trainingDataset = sparkSession.createDataFrame(t, schema)
        val validationDataset = sparkSession.createDataFrame(v, schema)
        val model = rfc.fit(trainingDataset)

        trainingDataset.unpersist()
        // rawPrediction == probabilityCol
        val val_result = model.transform(validationDataset)
          .drop(strategy.featuresCol).withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
        out_train = if (out_train == null) val_result else out_train.union(val_result)
        if (!isScan) {
          val val_acc = Evaluator.evaluate(val_result)
          train_metric += val_acc
          println(s"[$getNowTime] $message ${numFolds}_folds.train_$splitIndex = $val_acc")
        }
        validationDataset.unpersist()
    }
    (out_train, train_metric, rfc.fit(training))
  }

  def cvClassVectorGeneratorWithValidation(
      bcastTraining: Broadcast[DataFrame],
      bcastTesting: Broadcast[DataFrame],
      rfc: RandomForestCARTClassifier,
      numFolds: Int,
      seed: Long,
      strategy: GCForestStrategy,
      isScan: Boolean,
      message: String):
  (DataFrame, DataFrame, Metric, Metric, Array[RandomForestCARTModel]) = {
    val schema = bcastTraining.value.schema
    val sparkSession = bcastTraining.value.sparkSession
    var out_train: DataFrame = null // closure need
    var out_test: DataFrame = null // closure need

    require(bcastTraining.value.schema.equals(bcastTesting.value.schema))

    // cross-validation for k classes distribution features
    var train_metric = new Accuracy(0, 0)
    val splits = MLUtils.kFold(bcastTraining.value.toDF.rdd, numFolds, seed * System.currentTimeMillis())
    val models = splits.zipWithIndex.map {
      case ((t, v), splitIndex) =>
        val trainingDataset = sparkSession.createDataFrame(t, schema)
        val validationDataset = sparkSession.createDataFrame(v, schema)
        val model = rfc.fit(trainingDataset)

        trainingDataset.unpersist()
        // rawPrediction == probabilityCol
        val val_result = model.transform(validationDataset).drop(strategy.featuresCol)
            .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
        out_train = if (out_train == null) val_result else out_train.union(val_result)

        if (!isScan) {
          val val_acc = Evaluator.evaluate(val_result)
          train_metric += val_acc
          // println(s"train_metric += $val_acc")
          println(s"[$getNowTime] $message ${numFolds}_folds.train_$splitIndex = $val_acc")
        }

        val test_result = model.transform(bcastTesting.value)
          .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol+s"$splitIndex")
          .select(strategy.instanceCol, strategy.labelCol, strategy.featuresCol+s"$splitIndex")
        out_test = if (out_test == null) test_result
          else out_test.join(test_result, Seq(strategy.instanceCol, strategy.labelCol))

        validationDataset.unpersist()
        model
    }
    out_test = out_test.withColumn(strategy.featuresCol,
      UDF.mergeVectorForKfold(3)(Range(0, numFolds).map(k => col(strategy.featuresCol + s"$k")): _*))
      .select(strategy.instanceCol, strategy.labelCol, strategy.featuresCol)
    val test_metric = if (!isScan) Evaluator.evaluate(out_test) else new Accuracy(0, 0)
    (out_train, out_test, train_metric, test_metric, models)
  }

  // create a random forest classifier by type
  def genRFClassifier(rfType: String,
                      strategy: GCForestStrategy,
                      isScan: Boolean,
                      num: Int): RandomForestCARTClassifier = {
    val rf = rfType match {
      case "rfc" => new RandomForestCARTClassifier()
      case "crfc" => new CompletelyRandomForestClassifier()
    }

    rf.setNumTrees(if (isScan) strategy.scanForestTreeNum else strategy.cascadeForestTreeNum)
      .setMaxBins(strategy.maxBins)
      .setMaxDepth(strategy.maxDepth)
      .setMinInstancesPerNode(if (isScan) strategy.scanMinInsPerNode else strategy.cascadeMinInsPerNode)
      .setFeatureSubsetStrategy("sqrt")
      .setCacheNodeIds(strategy.cacheNodeId)
      .setSeed(System.currentTimeMillis() + num*123L + rfType.hashCode % num)
  }

  /**
    * Concat multi-scan features
    * @param dataset one of a window
    * @param sets the others
    * @return input for Cascade Forest
    */
  def concatenate(
      strategy: GCForestStrategy,
      dataset: Dataset[_],
      sets: Dataset[_]*
     ): DataFrame = {
    val sparkSession = dataset.sparkSession
    var unionSet = dataset.toDF
    sets.foreach(ds => unionSet = unionSet.union(ds.toDF))

    class Record(val instance: Long,    // instance id
                 val label: Double,     // label
                 val features: Vector,  // features
                 val scanId: Int,       // the scan id for multi-scan
                 val forestId: Int,     // forest id
                 val winId: Long) extends Serializable // window id

    val concatData = unionSet.select(
      strategy.instanceCol, strategy.labelCol,
      strategy.featuresCol, strategy.scanCol,
      strategy.forestIdCol, strategy.winCol).rdd.map {
      row =>
        val instance = row.getAs[Long](strategy.instanceCol)
        val label = row.getAs[Double](strategy.labelCol)
        val features = row.getAs[Vector](strategy.featuresCol)
        val scanId = row.getAs[Int](strategy.scanCol)
        val forestId = row.getAs[Int](strategy.forestIdCol)
        val winId = row.getAs[Long](strategy.winCol)

        new Record(instance, label, features, scanId, forestId, winId)
    }.groupBy(
      record => record.instance
    ).map { group =>
      val instance = group._1
      val records = group._2
      val label = records.head.label

      def recordCompare(left: Record, right: Record): Boolean = {
        var code = left.scanId.compareTo(right.scanId)
        if (code == 0) code = left.forestId.compareTo(right.forestId)
        if (code == 0) code = left.winId.compareTo(right.winId)
        code < 0
      }

      val features = new DenseVector(records.toSeq.sortWith(recordCompare)
        .flatMap(_.features.toArray).toArray)
      // features = [0, 0, ..., 0] (903 dim)
      Row.fromSeq(Array[Any](instance, label, features))
    }

    val schema: StructType = StructType(Seq[StructField]())
      .add(StructField(strategy.instanceCol, LongType))
      .add(StructField(strategy.labelCol, DoubleType))
      .add(StructField(strategy.featuresCol, new VectorUDT))
    sparkSession.createDataFrame(concatData, schema)
  }

  /**
    * concat inputs of Cascade Forest with prediction
    * @param feature input features
    * @param predict prediction features
    * @return
    */
  def mergeFeatureAndPredict(
      feature: Dataset[_],
      predict: Dataset[_],
      strategy: GCForestStrategy): DataFrame = {
    val vectorMerge = udf { (v1: Vector, v2: Vector) =>
      new DenseVector(v1.toArray.union(v2.toArray))
    }

    if (predict != null) {
      feature.join(
        // join (predict feature col to predictionCol)
        predict.withColumnRenamed(strategy.featuresCol, strategy.predictionCol),
        Seq(strategy.instanceCol)  // join on instanceCol
        // add a featureCol with featureCol + predictionCol
      ).withColumn(strategy.featuresCol, vectorMerge(col(strategy.featuresCol),
        col(strategy.predictionCol))
      ).select(strategy.instanceCol, strategy.featuresCol, strategy.labelCol).toDF
      // select 3 cols to DataFrame
    } else {
      feature.toDF
    }
  }

  private def getNowTime = dateFormat.format(new Date())

  /**
    *  Multi-Grained Scanning
    */
  def multi_grain_Scan(
      dataset: Dataset[_],
      strategy: GCForestStrategy): (DataFrame, Array[MultiGrainedScanModel]) = {

    require(dataset != null, "Null dataset need not to scan")

    var scanFeature: DataFrame = null
    val mgsModels = ArrayBuffer[MultiGrainedScanModel]()
    val rand = new Random()
    rand.setSeed(System.currentTimeMillis())

    println(s"[$getNowTime] Multi Grained Scanning begin!")

    if (strategy.dataStyle == "Img" && strategy.multiScanWindow.length > 0) {
      require(strategy.multiScanWindow.length % 2 == 0,
        "The multiScanWindow must has the even number for image-style data")

      val scanFeatures = ArrayBuffer[Dataset[_]]()

      Range(0, strategy.multiScanWindow.length / 2).foreach { i =>
        // Get the size of scan window
        val (w, h) = (strategy.multiScanWindow(i), strategy.multiScanWindow(i+1))
        val windowInstances = extractMatrixRDD(dataset, w, h, strategy.dataSize, strategy.featuresCol, strategy.winCol)

        val rfc =
          genRFClassifier("rfc", strategy, isScan = true, 0)
        var (rfcFeature, _, rfcModel) =
          cvClassVectorGenerator(windowInstances, rfc, strategy.numFolds, strategy.seed, strategy,
            isScan = true, "Scan 1")
        rfcFeature = rfcFeature.withColumn(strategy.forestIdCol, lit(1)).withColumn(strategy.scanCol, lit(i))

        scanFeatures += rfcFeature

        val crfc =
          genRFClassifier("crfc", strategy, isScan = true, 1)
        var (crfcFeature, _, crfcModel) =
          cvClassVectorGenerator(windowInstances, crfc, strategy.numFolds, strategy.seed, strategy,
            isScan = true, "Scan 2")
        crfcFeature = crfcFeature.withColumn(strategy.forestIdCol, lit(2)).withColumn(strategy.scanCol, lit(i))

        scanFeatures += crfcFeature

        mgsModels += new MultiGrainedScanModel(Array(w, h), rfcModel, crfcModel)
      }
      scanFeature =
        concatenate(strategy, scanFeatures.head, scanFeatures.tail:_*).cache

    } else if (strategy.dataStyle == "Seq" && strategy.multiScanWindow.length > 0) {
      val scanFeatures = ArrayBuffer[Dataset[_]]()
      strategy.multiScanWindow.indices.foreach { i => // for each window
        val windowSize = strategy.multiScanWindow(i)
        val windowInstances = extractSequenceRDD(dataset, windowSize, strategy.dataSize,
          strategy.featuresCol, strategy.winCol)

        val rfc =
          genRFClassifier("rfc", strategy, isScan = true, 0)
        var (rfcFeature, _, rfcModel) =
          cvClassVectorGenerator(windowInstances, rfc, strategy.numFolds, strategy.seed, strategy,
            isScan = true, "Scan 1")
        rfcFeature = rfcFeature.withColumn(strategy.forestIdCol, lit(1)).withColumn(strategy.scanCol, lit(i))

        scanFeatures += rfcFeature

        val crfc =
          genRFClassifier("crfc", strategy, isScan = true, 1)
        var (crfcFeature, _, crfcModel) =
          cvClassVectorGenerator(windowInstances, crfc, strategy.numFolds, strategy.seed, strategy,
            isScan = true, "Scan 2")
        crfcFeature = crfcFeature.withColumn(strategy.forestIdCol, lit(2)).withColumn(strategy.scanCol, lit(i))
        scanFeatures += crfcFeature

        mgsModels += new MultiGrainedScanModel(Array(windowSize), rfcModel, crfcModel)
      }
      scanFeature =
        concatenate(strategy, scanFeatures.head, scanFeatures.tail:_*).cache

    } else if (strategy.multiScanWindow.length > 0){
      throw new UnsupportedOperationException(s"The dataStyle: ${strategy.dataStyle} is unsupported!")
    }

    if (strategy.multiScanWindow.length == 0)
      scanFeature = dataset.toDF
    // scanFeature: (instanceId, label, features)
    println(s"[$getNowTime] Multi Grained Scanning finished!")
    (scanFeature, mgsModels.toArray)
  }

  def train(
      input: Dataset[_],
      strategy: GCForestStrategy): GCForestClassificationModel = {
    val numClasses: Int = strategy.classNum
    val erfModels = ArrayBuffer[Array[RandomForestCARTModel]]()
    val n_train = input.count()

    val (scanFeature_train, mgsModels) = multi_grain_Scan(input, strategy)

    scanFeature_train.cache()

    println(s"[$getNowTime] Cascade Forest begin!")

    val sparkSession = scanFeature_train.sparkSession
    val sc = sparkSession.sparkContext
    val rng = new Random()
    rng.setSeed(System.currentTimeMillis())

    var lastPrediction: DataFrame = null
    val acc_list = ArrayBuffer[Double]()

    // Init classifiers
    val maxIteration = strategy.maxIteration
    require(maxIteration > 0, "Non-positive maxIteration")
    var layer_id = 1
    var reachMaxLayer = false

    while (!reachMaxLayer) {

      println(s"[$getNowTime] Training Cascade Forest Layer $layer_id")

      val randomForests = (Range(0, 4).map ( it => genRFClassifier("rfc", strategy, isScan = false, rng.nextInt + it))
        ++
        Range(4, 8).map ( it => genRFClassifier("crfc", strategy, isScan = false, rng.nextInt + it))
      ).toArray[RandomForestCARTClassifier]
      assert(randomForests.length == 8, "random Forests inValid!")
      // scanFeatures_*: (instanceId, label, features)
      val training = mergeFeatureAndPredict(scanFeature_train, lastPrediction, strategy)
        .persist(StorageLevel.MEMORY_ONLY_SER)
      val bcastTraining = sc.broadcast(training)
      val features_dim = training.first().mkString.split(",").length

      println(s"[$getNowTime] Training Set = ($n_train, $features_dim)")

      var ensemblePredict: DataFrame = null  // closure need

      var layer_train_metric: Accuracy = new Accuracy(0, 0)  // closure need

      println(s"[$getNowTime] Forests fitting and transforming ......")

      erfModels += randomForests.zipWithIndex.map { case (rf, it) =>
        val transformed = cvClassVectorGenerator(
          bcastTraining.value, rf, strategy.numFolds, strategy.seed, strategy,
          isScan = false, s"layer [$layer_id] - estimator [$it]")
        val predict = transformed._1
          .withColumn(strategy.forestIdCol, lit(it))
          .select(strategy.instanceCol, strategy.featuresCol, strategy.forestIdCol)

        ensemblePredict =
          if (ensemblePredict == null) predict else ensemblePredict.union(predict)

        layer_train_metric = layer_train_metric + transformed._2

        println(s"[$getNowTime] [Estimator Summary] " +
          s"layer [$layer_id] - estimator [$it] Train.predict = ${transformed._2}")
        transformed._3
      }

      println(s"[$getNowTime] [Layer Summary] layer [$layer_id] - " +
        s"train.classifier.average = ${layer_train_metric.div(8d)}")
      println(s"[$getNowTime] Forests fitting and transforming finished!")

      val schema = new StructType()
        .add(StructField(strategy.instanceCol, LongType))
        .add(StructField(strategy.featuresCol, new VectorUDT))

      println(s"[$getNowTime] Getting prediction RDD ......")

      acc_list += layer_train_metric.getAccuracy

      val predictRDDs = {
        val grouped = ensemblePredict.rdd.groupBy(_.getAs[Long](strategy.instanceCol))
        grouped.map { group =>
          val instanceId = group._1
          val rows = group._2
          val features = new DenseVector(rows.toArray
            .sortWith(_.getAs[Int](strategy.forestIdCol) < _.getAs[Int](strategy.forestIdCol))
            .flatMap(_.getAs[Vector](strategy.featuresCol).toArray))
          Row.fromSeq(Array[Any](instanceId, features))
        }
      }
      //predictRDDs.foreach(r => r.persist(StorageLevel.MEMORY_ONLY_SER))
      println(s"[$getNowTime] Get prediction RDD finished! Layer $layer_id training finished!")

      val opt_layer_id_train = acc_list.zipWithIndex.maxBy(_._1)._2


      if (opt_layer_id_train + 1 == layer_id)
        println(s"[$getNowTime] [Result] [Optimal Layer] max_layer_num = $layer_id " +
          s"accuracy_train = ${acc_list(opt_layer_id_train)*100}%")

      lastPrediction = sparkSession.createDataFrame(predictRDDs, schema).cache()
      val outOfRounds = layer_id - opt_layer_id_train >= strategy.earlyStoppingRounds
      if (outOfRounds)
        println(s"[$getNowTime] " +
          s"[Result][Optimal Level Detected] opt_layer_id = " +
          s"$opt_layer_id_train, " +
          s"accuracy_train=${acc_list(opt_layer_id_train)}")
      reachMaxLayer = (layer_id == maxIteration) || outOfRounds
      if (reachMaxLayer)
        println(s"[$getNowTime] " +
          s"[Result][Reach Max Layer] max_layer_num=$layer_id, " +
          s"accuracy_train=$layer_train_metric")
      layer_id += 1
    }

    scanFeature_train.unpersist

    println(s"[$getNowTime] Cascade Forest Training Finished!")

    new GCForestClassificationModel(mgsModels, erfModels.toArray, numClasses)
  }

  /**
    *  Train a Cascade Forest
    */
  private def trainWithValidation(
              input: Dataset[_],
              validationInput: Dataset[_],
              strategy: GCForestStrategy): GCForestClassificationModel = {
    val numClasses: Int = strategy.classNum
    val erfModels = ArrayBuffer[Array[RandomForestCARTModel]]() // layer - (forest * fold)
    val n_train = input.count()
    val n_test = validationInput.count()

    val (scanFeature_train, mgsModels) = multi_grain_Scan(input, strategy)
    val (scanFeature_test, mgsModels_test) = multi_grain_Scan(validationInput, strategy)

    scanFeature_train.cache()
    scanFeature_test.cache()

    println(s"[$getNowTime] Cascade Forest begin!")

    val sparkSession = scanFeature_train.sparkSession
    val sc = sparkSession.sparkContext
    val rng = new Random()
    rng.setSeed(System.currentTimeMillis())

    var lastPrediction: DataFrame = null
    var lastPrediction_test: DataFrame = null
    val acc_list = Array(ArrayBuffer[Double](), ArrayBuffer[Double]())

    // Init classifiers
    val maxIteration = strategy.maxIteration
    require(maxIteration > 0, "Non-positive maxIteration")
    var layer_id = 1
    var reachMaxLayer = false
    val bcastStrategy = sc.broadcast(strategy)

    while (!reachMaxLayer) {

      println(s"[$getNowTime] Training Cascade Forest Layer $layer_id")

      val randomForests = (Range(0, 4).map ( it => genRFClassifier("rfc", strategy, isScan = false, rng.nextInt + it))
        ++
        Range(4, 8).map ( it => genRFClassifier("crfc", strategy, isScan = false, rng.nextInt + it))
        ).toArray[RandomForestCARTClassifier]
      assert(randomForests.length == 8, "random Forests inValid!")
      // scanFeatures_*: (instanceId, label, features)
      val training = mergeFeatureAndPredict(scanFeature_train, lastPrediction, strategy)
        .repartition(sc.defaultParallelism)
        .persist(StorageLevel.MEMORY_AND_DISK_SER)
      val testing = mergeFeatureAndPredict(scanFeature_test, lastPrediction_test, strategy)
        .repartition(sc.defaultParallelism)
        .persist(StorageLevel.MEMORY_AND_DISK_SER)
      val bcastTraining = sc.broadcast(training)
      val bcastTesting = sc.broadcast(testing)
      val features_dim = training.first().mkString.split(",").length

      println(s"[$getNowTime] Training Set = ($n_train, $features_dim), " +
        s"Testing Set = ($n_test, $features_dim)")

      if (lastPrediction != null) lastPrediction.unpersist(blocking = false)
      if (lastPrediction_test != null) lastPrediction_test.unpersist(blocking = false)

      var ensemblePredict: DataFrame = null  // closure need
      var ensemblePredict_test: DataFrame = null  // closure need

      var layer_train_metric: Accuracy = new Accuracy(0, 0)  // closure need
      var layer_test_metric: Accuracy = new Accuracy(0, 0)  // closure need

      println(s"[$getNowTime] Forests fitting and transforming ......")

      erfModels ++= randomForests.zipWithIndex.map { case (rf, it) =>
        val st = bcastStrategy.value
        val transformed = cvClassVectorGeneratorWithValidation(
          bcastTraining, bcastTesting, rf, st.numFolds, st.seed, st,
          isScan = false, s"layer [$layer_id] - estimator [$it]")
        val predict = transformed._1
          .withColumn(st.forestIdCol, lit(it))
          .select(st.instanceCol, st.featuresCol, st.forestIdCol)
        val predict_test = transformed._2
          .withColumn(st.forestIdCol, lit(it))
          .select(st.instanceCol, st.featuresCol, st.forestIdCol)
        ensemblePredict =
          if (ensemblePredict == null) predict else ensemblePredict.union(predict)
        ensemblePredict_test =
          if (ensemblePredict_test == null) predict_test else ensemblePredict_test
            .union(predict_test)

        layer_train_metric = layer_train_metric + transformed._3
        // println(s"layer_train_metric RF_$it add ${transformed._3}")
        layer_test_metric = layer_test_metric + transformed._4
        // println(s"layer_test_metric RF_$it add ${transformed._4}")

        println(s"[$getNowTime] [Estimator Summary] " +
          s"layer [$layer_id] - estimator [$it] Train.predict = ${transformed._3}")
        println(s"[$getNowTime] [Estimator Summary] " +
          s"layer [$layer_id] - estimator [$it]  Test.predict = ${transformed._4}")
        if (st.idebug) {
          println("Model ==========================================")
          println("total Number of Nodes: " + transformed._5.map(_.totalNumNodes).mkString(" , "))
          println("First Tree Structure: " + transformed._5(0).trees(0).toDebugString)
          println("Model ==========================================")
        }
        transformed._5
      }

      println(s"[$getNowTime] [Layer Summary] layer [$layer_id] - " +
        s"train.classifier.average = ${layer_train_metric.div(8d)}")
      println(s"[$getNowTime] [Layer Summary] layer [$layer_id] - " +
        s"test.classifier.average = ${layer_test_metric.div(8d)}")
      println(s"[$getNowTime] Forests fitting and transforming finished!")

      acc_list(0) += layer_train_metric.getAccuracy
      acc_list(1) += layer_test_metric.getAccuracy

      val schema = new StructType()
        .add(StructField(bcastStrategy.value.instanceCol, LongType))
        .add(StructField(bcastStrategy.value.featuresCol, new VectorUDT))

      println(s"[$getNowTime] Getting prediction RDD ......")

      val predictRDDs =
        Array(ensemblePredict, ensemblePredict_test).map { predict =>
          val grouped = predict.rdd.groupBy(_.getAs[Long](bcastStrategy.value.instanceCol))
          //          println(s"grouped $idx partition: ${grouped.getNumPartitions}")
          val predictRDD = grouped.map { group =>
            val instanceId = group._1
            val rows = group._2
            val features = new DenseVector(rows.toArray
              .sortWith(_.getAs[Int](bcastStrategy.value.forestIdCol) < _.getAs[Int](bcastStrategy.value.forestIdCol))
              .flatMap(_.getAs[Vector](bcastStrategy.value.featuresCol).toArray))
            Row.fromSeq(Array[Any](instanceId, features))
          }
          predictRDD
        }
      //predictRDDs.foreach(r => r.persist(StorageLevel.MEMORY_ONLY_SER))
      println(s"[$getNowTime] Get prediction RDD finished! Layer $layer_id training finished!")

      val opt_layer_id_train = acc_list(0).zipWithIndex.maxBy(_._1)._2
      val opt_layer_id_test = acc_list(1).zipWithIndex.maxBy(_._1)._2

      if (strategy.earlyStopByTest) {
        if (opt_layer_id_test + 1 == layer_id)
          println(s"[$getNowTime] [Result] [Optimal Layer] max_layer_num = $layer_id " +
            "accuracy_train=%.3f %%, ".format(acc_list(0)(opt_layer_id_train)*100) +
            "accuracy_test=%.3f %%".format(acc_list(1)(opt_layer_id_test)*100))
      }
      else {
        if (opt_layer_id_train + 1 == layer_id)
          println(s"[$getNowTime] [Result] [Optimal Layer] max_layer_num = $layer_id " +
            "accuracy_train=%.3f %%, ".format(acc_list(0)(opt_layer_id_train)*100) +
            "accuracy_test=%.3f %%".format(acc_list(1)(opt_layer_id_test)*100))
      }

      lastPrediction = sparkSession.createDataFrame(predictRDDs(0), schema).persist(StorageLevel.MEMORY_AND_DISK_SER)
      lastPrediction_test = sparkSession.createDataFrame(predictRDDs(1), schema)
        .persist(StorageLevel.MEMORY_AND_DISK_SER)
      val outOfRounds =
        (strategy.earlyStopByTest && layer_id - opt_layer_id_test >= strategy.earlyStoppingRounds) ||
        (!strategy.earlyStopByTest && layer_id - opt_layer_id_train >= strategy.earlyStoppingRounds)
      if (outOfRounds)
        println(s"[$getNowTime] " +
          s"[Result][Optimal Level Detected] opt_layer_id = " +
          s"${if (strategy.earlyStopByTest) opt_layer_id_test else opt_layer_id_train}, " +
          "accuracy_train=%.3f %%, ".format(acc_list(0)(opt_layer_id_train)*100) +
          "accuracy_test=%.3f %%".format(acc_list(1)(opt_layer_id_test)*100))
      reachMaxLayer = (layer_id == maxIteration) || outOfRounds
      if (reachMaxLayer)
        println(s"[$getNowTime] " +
          s"[Result][Reach Max Layer] max_layer_num=$layer_id, " +
          s"accuracy_train=$layer_train_metric, accuracy_test=$layer_test_metric")
      layer_id += 1
      bcastTraining.destroy(blocking = false)
      bcastTesting.destroy(blocking = false)
      training.unpersist(blocking = false)
      testing.unpersist(blocking = false)
    }
    bcastStrategy.destroy(blocking = false)
    scanFeature_train.unpersist
    scanFeature_test.unpersist
    println(s"[$getNowTime] Cascade Forest Training Finished!")

    new GCForestClassificationModel(mgsModels ++ mgsModels_test, erfModels.toArray, numClasses)
  }
}
