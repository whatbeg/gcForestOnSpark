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
import org.apache.spark.ml.tree.impl.GCForestImpl
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

import scala.collection.mutable.ArrayBuffer


class GCForestClassifier(override val uid: String)
  extends ProbabilisticClassifier[Vector, GCForestClassifier, GCForestClassificationModel]
  with DefaultParamsWritable with GCForestParams {

  def this() = this(Identifiable.randomUID("gcf"))

  override def setNumClasses(value: Int): this.type = set(classNum, value)

  override def setDataSize(value: Array[Int]): this.type = set(dataSize, value)

  override def setDataStyle(value: String): this.type = set(dataStyle, value)

  override def setMultiScanWindow(value: Array[Int]): this.type = set(multiScanWindow, value)

  override def setScanForestTreeNum(value: Int): this.type = set(scanForestTreeNum, value)

  override def setCascadeForestTreeNum(value: Int): this.type = set(cascadeForestTreeNum, value)

  override def setMaxIteration(value: Int): this.type = set(MaxIteration, value)

  override def setEarlyStoppingRounds(value: Int): this.type = set(earlyStoppingRounds, value)

  def getGCForestStrategy: GCForestStrategy = {
    GCForestStrategy($(classNum), $(multiScanWindow), $(dataSize), $(scanForestTreeNum),
      $(cascadeForestTreeNum), $(scanForestMinInstancesPerNode),
      $(cascadeForestMinInstancesPerNode), $(randomForestMaxBins), $(randomForestMaxDepth),
      $(MaxIteration), $(numFolds), $(earlyStoppingRounds), $(earlyStopByTest), $(dataStyle),
      $(seed), $(windowCol), $(scanCol), $(forestIdCol))
  }

  def getDefaultStrategy: GCForestStrategy = {
    GCForestStrategy(2, Array(), Array(113))
  }

  def train(trainset: Dataset[_], testset: Dataset[_]): GCForestClassificationModel = {
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

    copyValues(GCForestImpl.runWithValidation(casted_train, casted_test, getGCForestStrategy))
  }

  override def train(dataset: Dataset[_]): GCForestClassificationModel = {
    // This handles a few items such as schema validation.
    // Developers only need to implement train().
    transformSchema(dataset.schema, logging = true)

    // Cast LabelCol to DoubleType and keep the metadata.
    val labelMeta = dataset.schema($(labelCol)).metadata
    val casted_train =
      dataset.withColumn($(labelCol), col($(labelCol)).cast(DoubleType), labelMeta)
    GCForestImpl.run(casted_train, getGCForestStrategy)
  }

  override def copy(extra: ParamMap): GCForestClassifier = defaultCopy(extra)
}


private[ml] class GCForestClassificationModel (
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