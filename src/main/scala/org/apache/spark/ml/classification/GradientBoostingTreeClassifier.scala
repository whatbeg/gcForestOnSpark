/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.classification

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._

import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.GradientBoostedTrees
import org.apache.spark.ml.util._
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.mllib.tree.model.{GradientBoostedTreesModel => OldGBTModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._


/**
 * Gradient-Boosted Trees (GBTs) (http://en.wikipedia.org/wiki/Gradient_boosting)
 * learning algorithm for classification.
 * It supports binary labels, as well as both continuous and categorical features.
 *
 * The implementation is based upon: J.H. Friedman. "Stochastic Gradient Boosting." 1999.
 *
 * Notes on Gradient Boosting vs. TreeBoost:
 *  - This implementation is for Stochastic Gradient Boosting, not for TreeBoost.
 *  - Both algorithms learn tree ensembles by minimizing loss functions.
 *  - TreeBoost (Friedman, 1999) additionally modifies the outputs at tree leaf nodes
 *    based on the loss function, whereas the original gradient boosting method does not.
 *  - We expect to implement TreeBoost in the future:
 *    [https://issues.apache.org/jira/browse/SPARK-4240]
 *
 * @note Multiclass labels are not currently supported.
 */
@Since("1.4.0")
class GradientBoostingTreeClassifier (override val uid: String)
  extends ProbabilisticClassifier[Vector, GradientBoostingTreeClassifier,
    GradientBoostingClassificationModel]
    with GBTClassifierParams with DefaultParamsWritable with Logging {

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("gbtc"))

  // Override parameter setters from parent trait for Java API compatibility.

  // Parameters from TreeClassifierParams:

  /** @group setParam */
  @Since("1.4.0")
  override def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setMaxBins(value: Int): this.type = set(maxBins, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setMinInstancesPerNode(value: Int): this.type = set(minInstancesPerNode, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setMinInfoGain(value: Double): this.type = set(minInfoGain, value)

  /** @group expertSetParam */
  @Since("1.4.0")
  override def setMaxMemoryInMB(value: Int): this.type = set(maxMemoryInMB, value)

  /** @group expertSetParam */
  @Since("1.4.0")
  override def setCacheNodeIds(value: Boolean): this.type = set(cacheNodeIds, value)

  /**
    * Specifies how often to checkpoint the cached node IDs.
    * E.g. 10 means that the cache will get checkpointed every 10 iterations.
    * This is only used if cacheNodeIds is true and if the checkpoint directory is set in
    * [[org.apache.spark.SparkContext]].
    * Must be at least 1.
    * (default = 10)
    * @group setParam
    */
  @Since("1.4.0")
  override def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  /**
    * The impurity setting is ignored for GBT models.
    * Individual trees are built using impurity "Variance."
    *
    * @group setParam
    */
  @Since("1.4.0")
  override def setImpurity(value: String): this.type = {
    logWarning("GradientBoostingTreeClassifier.setImpurity should NOT be used")
    this
  }

  // Parameters from TreeEnsembleParams:

  /** @group setParam */
  @Since("1.4.0")
  override def setSubsamplingRate(value: Double): this.type = set(subsamplingRate, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setSeed(value: Long): this.type = set(seed, value)

  // Parameters from GBTParams:

  /** @group setParam */
  @Since("1.4.0")
  override def setMaxIter(value: Int): this.type = set(maxIter, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setStepSize(value: Double): this.type = set(stepSize, value)

  // Parameters from GradientBoostingTreeClassifierParams:

  /** @group setParam */
  @Since("1.4.0")
  def setLossType(value: String): this.type = set(lossType, value)

  override protected def train(dataset: Dataset[_]): GradientBoostingClassificationModel = {
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    // We copy and modify this from Classifier.extractLabeledPoints since GBT only supports
    // 2 classes now.  This lets us provide a more precise error message.
    val oldDataset: RDD[LabeledPoint] =
    dataset.select(col($(labelCol)), col($(featuresCol))).rdd.map {
      case Row(label: Double, features: Vector) =>
        require(label == 0 || label == 1, s"GradientBoostingTreeClassifier was given" +
          s" dataset with invalid label $label.  Labels must be in {0,1}; note that" +
          s" GradientBoostingTreeClassifier currently only supports binary classification.")
        LabeledPoint(label, features)
    }
    val numFeatures = oldDataset.first().features.size
    val boostingStrategy = super.getOldBoostingStrategy(categoricalFeatures, OldAlgo.Classification)

    val numClasses = 2
    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".train() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }

    val instr = Instrumentation.create(this, oldDataset)
    instr.logParams(labelCol, featuresCol, predictionCol, impurity, lossType,
      maxDepth, maxBins, maxIter, maxMemoryInMB, minInfoGain, minInstancesPerNode,
      seed, stepSize, subsamplingRate, cacheNodeIds, checkpointInterval)
    instr.logNumFeatures(numFeatures)
    instr.logNumClasses(numClasses)

    val (baseLearners, learnerWeights) = GradientBoostedTrees.run(oldDataset, boostingStrategy,
      $(seed))
    val m = new GradientBoostingClassificationModel(uid, baseLearners, learnerWeights, numFeatures)
    instr.logSuccess(m)
    m
  }

  @Since("1.4.1")
  override def copy(extra: ParamMap): GradientBoostingTreeClassifier = defaultCopy(extra)
}

@Since("1.4.0")
object GradientBoostingTreeClassifier
  extends DefaultParamsReadable[GradientBoostingTreeClassifier] {

  /** Accessor for supported loss settings: logistic */
  @Since("1.4.0")
  final val supportedLossTypes: Array[String] = GBTClassifierParams.supportedLossTypes

  @Since("2.0.0")
  override def load(path: String): GradientBoostingTreeClassifier = super.load(path)
}

/**
 * Gradient-Boosted Trees (GBTs) (http://en.wikipedia.org/wiki/Gradient_boosting)
 * model for classification.
 * It supports binary labels, as well as both continuous and categorical features.
 *
 * @param _trees  Decision trees in the ensemble.
 * @param _treeWeights  Weights for the decision trees in the ensemble.
 *
 * @note Multiclass labels are not currently supported.
 */
@Since("1.6.0")
class GradientBoostingClassificationModel private[ml](
      @Since("1.6.0") override val uid: String,
      private val _trees: Array[DecisionTreeRegressionModel],
      private val _treeWeights: Array[Double],
      @Since("1.6.0") override val numFeatures: Int,
      @Since("2.2.0") override val numClasses: Int)
  extends ProbabilisticClassificationModel[Vector, GradientBoostingClassificationModel]
    with GBTClassifierParams with TreeEnsembleModel[DecisionTreeRegressionModel]
    with MLWritable with Serializable {

  require(_trees.nonEmpty, "GradientBoostingClassificationModel requires at least 1 tree.")
  require(_trees.length == _treeWeights.length,
    "GradientBoostingClassificationModel given trees, treeWeights" +
    s" of non-matching lengths (${_trees.length}, ${_treeWeights.length}, respectively).")

  /**
    * Construct a GradientBoostingClassificationModel
    *
    * @param _trees  Decision trees in the ensemble.
    * @param _treeWeights  Weights for the decision trees in the ensemble.
    * @param numFeatures  The number of features.
    */
  private[ml] def this(
                  uid: String,
                  _trees: Array[DecisionTreeRegressionModel],
                  _treeWeights: Array[Double],
                  numFeatures: Int) =
    this(uid, _trees, _treeWeights, numFeatures, 2)

  /**
    * Construct a GradientBoostingClassificationModel
    *
    * @param _trees  Decision trees in the ensemble.
    * @param _treeWeights  Weights for the decision trees in the ensemble.
    */
  @Since("1.6.0")
  def this(uid: String, _trees: Array[DecisionTreeRegressionModel], _treeWeights: Array[Double]) =
  this(uid, _trees, _treeWeights, -1, 2)

  @Since("1.4.0")
  override def trees: Array[DecisionTreeRegressionModel] = _trees

  /**
    * Number of trees in ensemble
    */
  @Since("2.0.0")
  val getNumTrees: Int = trees.length

  @Since("1.4.0")
  override def treeWeights: Array[Double] = _treeWeights

  override protected def transformImpl(dataset: Dataset[_]): DataFrame = {
    val bcastModel = dataset.sparkSession.sparkContext.broadcast(this)
    val predictUDF = udf { (features: Any) =>
      bcastModel.value.predict(features.asInstanceOf[Vector])
    }
    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    val bcastModel = dataset.sparkSession.sparkContext.broadcast(this)
    val probabilityUDF = udf { (features: Any) =>
      bcastModel.value.predictProbability(features.asInstanceOf[Vector])
    }
    dataset.withColumn($(probabilityCol), probabilityUDF(col($(featuresCol))))
  }

  override protected def predict(features: Vector): Double = {
    // If thresholds defined, use predictRaw to get probabilities, otherwise use optimization
    if (isDefined(thresholds)) {
      super.predict(features)
    } else {
      if (margin(features) > 0.0) 1.0 else 0.0
    }
  }

  override protected def predictRaw(features: Vector): Vector = {
    val prediction: Double = margin(features)
    Vectors.dense(Array(-prediction, prediction))
  }

  override def predictProbability(features: Vector): Vector = super.predictProbability(features)

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        dv.values(0) = loss.computeProbability(dv.values(0))
        dv.values(1) = 1.0 - dv.values(0)
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in GradientBoostingClassificationModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  /** Number of trees in ensemble */
  val numTrees: Int = trees.length

  @Since("1.4.0")
  override def copy(extra: ParamMap): GradientBoostingClassificationModel = {
    copyValues(new GradientBoostingClassificationModel(
      uid, _trees, _treeWeights, numFeatures, numClasses), extra).setParent(parent)
  }

  @Since("1.4.0")
  override def toString: String = {
    s"GradientBoostingClassificationModel (uid=$uid) with $numTrees trees"
  }

  /**
    * Estimate of the importance of each feature.
    *
    * Each feature's importance is the average of its importance across all trees in the ensemble
    * The importance vector is normalized to sum to 1. This method is suggested by Hastie et al.
    * (Hastie, Tibshirani, Friedman. "The Elements of Statistical Learning, 2nd Edition." 2001.)
    * and follows the implementation from scikit-learn.

    * See `DecisionTreeClassificationModel.featureImportances`
    */
  @Since("2.0.0")
  lazy val featureImportances: Vector = TreeEnsembleModel.featureImportances(trees, numFeatures)

  /** Raw prediction for the positive class. */
  private def margin(features: Vector): Double = {
    val treePredictions = _trees.map(_.rootNode.predictImpl(features).prediction)
    blas.ddot(numTrees, treePredictions, 1, _treeWeights, 1)
  }

  /** (private[ml]) Convert to a model in the old API */
  private[ml] def toOld: OldGBTModel = {
    new OldGBTModel(OldAlgo.Classification, _trees.map(_.toOld), _treeWeights)
  }

  // hard coded loss, which is not meant to be changed in the model
  private val loss = getOldLossType

  @Since("2.0.0")
  override def write: MLWriter =
    new GradientBoostingClassificationModel.GradientBoostingClassificationModelWriter(this)
}

@Since("2.0.0")
object GradientBoostingClassificationModel extends MLReadable[GradientBoostingClassificationModel] {

  private val numFeaturesKey: String = "numFeatures"
  private val numTreesKey: String = "numTrees"

  @Since("2.0.0")
  override def read: MLReader[GradientBoostingClassificationModel] =
    new GradientBoostingClassificationModelReader

  @Since("2.0.0")
  override def load(path: String): GradientBoostingClassificationModel = super.load(path)

  private[ml]
  class GradientBoostingClassificationModelWriter(instance: GradientBoostingClassificationModel)
    extends MLWriter {

    override protected def saveImpl(path: String): Unit = {

      val extraMetadata: JObject = Map(
        numFeaturesKey -> instance.numFeatures,
        numTreesKey -> instance.getNumTrees)
      EnsembleModelReadWrite.saveImpl(instance, path, sparkSession, extraMetadata)
    }
  }

  private class GradientBoostingClassificationModelReader
    extends MLReader[GradientBoostingClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[GradientBoostingClassificationModel].getName
    private val treeClassName = classOf[DecisionTreeRegressionModel].getName

    override def load(path: String): GradientBoostingClassificationModel = {
      implicit val format = DefaultFormats
      val (metadata: Metadata, treesData: Array[(Metadata, Node)], treeWeights: Array[Double]) =
        EnsembleModelReadWrite.loadImpl(path, sparkSession, className, treeClassName)
      val numFeatures = (metadata.metadata \ numFeaturesKey).extract[Int]
      val numTrees = (metadata.metadata \ numTreesKey).extract[Int]

      val trees: Array[DecisionTreeRegressionModel] = treesData.map {
        case (treeMetadata, root) =>
          val tree =
            new DecisionTreeRegressionModel(treeMetadata.uid, root, numFeatures)
          DefaultParamsReader.getAndSetParams(tree, treeMetadata)
          tree
      }
      require(numTrees == trees.length,
        s"GradientBoostingClassificationModel.load expected $numTrees" +
        s" trees based on metadata but found ${trees.length} trees.")
      val model = new GradientBoostingClassificationModel(metadata.uid,
        trees, treeWeights, numFeatures)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

  /** Convert a model from the old API */
  private[ml] def fromOld(
                           oldModel: OldGBTModel,
                           parent: GradientBoostingTreeClassifier,
                           categoricalFeatures: Map[Int, Int],
                           numFeatures: Int = -1,
                           numClasses: Int = 2): GradientBoostingClassificationModel = {
    require(oldModel.algo == OldAlgo.Classification, "Cannot convert GradientBoostedTreesModel" +
      s" with algo=${oldModel.algo} (old API) to GradientBoostingClassificationModel (new API).")
    val newTrees = oldModel.trees.map { tree =>
      // parent for each tree is null since there is no good way to set this.
      DecisionTreeRegressionModel.fromOld(tree, null, categoricalFeatures)
    }
    val uid = if (parent != null) parent.uid else Identifiable.randomUID("gbtc")
    new GradientBoostingClassificationModel(
      uid, newTrees, oldModel.treeWeights, numFeatures, numClasses)
  }
}
