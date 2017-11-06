/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.tree.configuration

case class GCForestStrategy(
                            var classNum: Int = 2,
                            var multiScanWindow: Array[Int],
                            var dataSize: Array[Int],
                            var scanForestTreeNum: Int = 2,
                            var cascadeForestTreeNum: Int = 1,
                            var scanMinInsPerNode: Int = 1,
                            var cascadeMinInsPerNode: Int = 1,
                            var maxBins: Int = 32,
                            var maxDepth: Int = 30,
                            var maxIteration: Int = 2,
                            var numFolds: Int = 3,
                            var earlyStoppingRounds: Int = 4,
                            var earlyStopByTest: Boolean = true,
                            var dataStyle: String = "Seq",
                            var seed: Long = 123L,
                            var winCol: String = "windows",
                            var scanCol: String = "scan_id",
                            var forestIdCol: String = "forestNum",
                            var idebug: Boolean = false,

                            var instanceCol: String = "instance",
                            var rawPredictionCol: String = "rawPrediction",
                            var probabilityCol: String = "probability",
                            var predictionCol: String = "prediction",
                            var featuresCol: String = "features",
                            var labelCol: String = "label")
extends Serializable {

}
