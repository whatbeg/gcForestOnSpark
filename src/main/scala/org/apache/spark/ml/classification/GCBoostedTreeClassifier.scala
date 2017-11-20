/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.classification

import org.apache.hadoop.fs.Path
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.tree.GCForestParams
import org.apache.spark.ml.tree.configuration.GCForestStrategy
import org.apache.spark.ml.tree.impl.GCBoostedTreeImpl
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

import scala.collection.mutable.ArrayBuffer


class GCBoostedTreeClassifier(override val uid: String)
  extends ProbabilisticClassifier[Vector, GCBoostedTreeClassifier, GCBoostedTreeClassificationModel]
    with DefaultParamsWritable with GCForestParams {

  def this() = this(Identifiable.randomUID("gcbt"))

  override def setNumClasses(value: Int): this.type = set(classNum, value)

  override def setDataSize(value: Array[Int]): this.type = set(dataSize, value)

  override def setDataStyle(value: String): this.type = set(dataStyle, value)

  override def setMultiScanWindow(value: Array[Int]): this.type = set(multiScanWindow, value)

  override def setScanForestTreeNum(value: Int): this.type = set(scanForestTreeNum, value)

  override def setCascadeForestTreeNum(value: Int): this.type = set(cascadeForestTreeNum, value)

  override def setMaxIteration(value: Int): this.type = set(MaxIteration, value)

  override def setEarlyStoppingRounds(value: Int): this.type = set(earlyStoppingRounds, value)

  override def setIDebug(value: Boolean): GCBoostedTreeClassifier.this.type = set(idebug, value)

  override def setMaxDepth(value: Int): this.type = set(MaxDepth, value)

  override def setMaxBins(value: Int): GCBoostedTreeClassifier.this.type = set(MaxBins, value)

  override def setScanForestMinInstancesPerNode(value: Int):
  GCBoostedTreeClassifier.this.type = set(scanMinInsPerNode, value)

  override def setCascadeForestMinInstancesPerNode(value: Int):
  GCBoostedTreeClassifier.this.type = set(cascadeMinInsPerNode, value)

  override def setCacheNodeId(value: Boolean): GCBoostedTreeClassifier.this.type = set(cacheNodeId, value)

  override def setMaxMemoryInMB(value: Int): GCBoostedTreeClassifier.this.type = set(maxMemoryInMB, value)

  def getGCForestStrategy: GCForestStrategy = {
    GCForestStrategy($(classNum), $(modelPath), $(multiScanWindow), $(dataSize), $(rfNum), $(crfNum),
      $(scanForestTreeNum), $(cascadeForestTreeNum), $(scanMinInsPerNode), $(cascadeMinInsPerNode),
      $(MaxBins), $(MaxDepth), $(minInfoGain), $(MaxIteration), $(maxMemoryInMB), $(numFolds),
      $(earlyStoppingRounds), $(earlyStopByTest), $(dataStyle), $(seed), $(cacheNodeId), $(windowCol), $(scanCol),
      $(forestIdCol), $(idebug))
  }

  def getDefaultStrategy: GCForestStrategy = {
    GCForestStrategy(2, $(modelPath), Array(), Array(113), idebug = false)
  }

  def train(trainset: Dataset[_], testset: Dataset[_]): GCBoostedTreeClassificationModel = {
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

    copyValues(GCBoostedTreeImpl.runWithValidation(casted_train, casted_test, getGCForestStrategy))
  }

  override def train(dataset: Dataset[_]): GCBoostedTreeClassificationModel = {
    // This handles a few items such as schema validation.
    // Developers only need to implement train().
    transformSchema(dataset.schema, logging = true)

    // Cast LabelCol to DoubleType and keep the metadata.
    val labelMeta = dataset.schema($(labelCol)).metadata
    val casted_train =
      dataset.withColumn($(labelCol), col($(labelCol)).cast(DoubleType), labelMeta)
    GCBoostedTreeImpl.run(casted_train, getGCForestStrategy)
  }

  override def copy(extra: ParamMap): GCBoostedTreeClassifier = defaultCopy(extra)
}


private[ml] class GCBoostedTreeClassificationModel (
                  override val uid: String,
                  private val scanModel: Array[MultiGrainedScanModel],
                  private val cascadeForest: Array[Array[GradientBoostingClassificationModel]],
                  override val numClasses: Int)
  extends ProbabilisticClassificationModel[Vector, GCBoostedTreeClassificationModel]
    with GCForestParams with MLWritable with Serializable {

  def this(scanModel: Array[MultiGrainedScanModel],
           cascadeForest: Array[Array[GradientBoostingClassificationModel]],
           numClasses: Int) =
    this(Identifiable.randomUID("gcbtc"), scanModel, cascadeForest, numClasses)

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
    if ($(dataStyle) == "Img") {
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
    } else if ($(dataStyle) == "Seq") { // TODO
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
        throw new RuntimeException("Unexpected error in GCBoostedTreeClassificationModel:" +
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

  override def copy(extra: ParamMap): GCBoostedTreeClassificationModel = {
    copyValues(new GCBoostedTreeClassificationModel(uid,  scanModel, cascadeForest, numClasses), extra)
  }

  override def write: MLWriter =
    new GCBoostedTreeClassificationModel.GCBoostedTreeClassificationModelWriter(this)
}


object GCBoostedTreeClassificationModel extends MLReadable[GCBoostedTreeClassificationModel] {
  override def read: MLReader[GCBoostedTreeClassificationModel] = new GCBoostedTreeClassificationModelReader

  override def load(path: String): GCBoostedTreeClassificationModel = super.load(path)

  private[GCBoostedTreeClassificationModel]
  class GCBoostedTreeClassificationModelWriter(instance: GCBoostedTreeClassificationModel)
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

  private class GCBoostedTreeClassificationModelReader
    extends MLReader[GCBoostedTreeClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[GCBoostedTreeClassificationModel].getName
    val mgsClassName: String = classOf[MultiGrainedScanModel].getName

    override def load(path: String): GCBoostedTreeClassificationModel = {
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
          GradientBoostingClassificationModel.load(modelPath)
        }.toArray
      }.toArray

      val gcBoostedTreeModel =
        new GCBoostedTreeClassificationModel(gcMetadata.uid, scanModel, cascadeForest, numClasses)

      DefaultParamsReader.getAndSetParams(gcBoostedTreeModel, gcMetadata)
      gcBoostedTreeModel
    }
  }
}
