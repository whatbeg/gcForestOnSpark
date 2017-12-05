/**
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>

#define DBL_MAX (0x1.fffffffffffffP+1023)
#define DBL_MIN (-DBL_MAX)
#define SZD sizeof(double)

double calculateImpurity(char impurity, double allStats[], int statSize, int offset);
double* calcGainAndImpurityStats(char impurity, double ImpurityStats[], int statSize, int numSplits, double allStats[], int allStatsSize,
                                 int nodeFeatureOffset, int leftOffset, int minInsPerNode, double minInfoGain);
double* binToBestSplit(double ImpurityStats[], double allStats[], int featureOffset[], int nfeatureOffset,
                       int numSplits, char impurity, int statSize, int featureIndexIdx, int minInsPerNode, double minInfoGain);

double calculateImpurity(char impurity, double allStats[], int statSize, int offset) {
    if (impurity == 'g') { // gini
        double totalCount = 0;
        int i;
        for (i = 0; i < statSize; i++) {
            totalCount += allStats[offset + i];
        }
        if (totalCount <= 1e-9) return 0;
        double impurityResult = 1.0, freq;
        for (i = 0; i < statSize; i++) {
            freq = allStats[offset + i] / totalCount;
            impurityResult -= freq * freq;
        }
        return impurityResult;
    }
    else if (impurity == 'e') { // entropy
        return -1.0;
    }
    else if (impurity == 'v') { // variance
        return -1.0;
    }
    return -1.0;
}

/**
 * ImpurityStats Representation:
 * gain: Double
 * impurity: Double
 * impurityCalculator: Array[Double]
 * leftImpurityCalculator: Array[Double]
 * rightImpurityCalculator: Array[Double]
 * valid: double = 1 (-1 rep. false, 1 rep. true)
 *
 * thus, [gain, impurity, {_, _, _}, {_, _, _}, {_, _, _}, valid]
 */

double* calcGainAndImpurityStats(char impurity, double ImpurityStats[], int statSize, int numSplits, double allStats[], int allStatsSize, int nodeFeatureOffset, int leftOffset, int minInsPerNode, double minInfoGain) {
    double parentImpurityCalculator[statSize];
    double rightImpurityCalculator[statSize];
    double impurityResult = -1.0;
    memset(parentImpurityCalculator, 0, sizeof(parentImpurityCalculator));
    memset(rightImpurityCalculator, 0, sizeof(rightImpurityCalculator));
    if (ImpurityStats[2 + statSize * 3] < 0) { // stats == null
        memcpy(parentImpurityCalculator, allStats + nodeFeatureOffset + numSplits * statSize, statSize * SZD); // allStats to be pointer or cannot use this interface
        impurityResult = calculateImpurity(impurity, parentImpurityCalculator, statSize, 0);
    }
    else {
        memcpy(parentImpurityCalculator, ImpurityStats + 2, statSize * SZD);
        impurityResult = ImpurityStats[1];
    }
    double leftCount = 0, rightCount = 0;
    int i;
    for (i = 0; i < statSize; i++) {
        leftCount += allStats[leftOffset + i];
    }
    int lastOffset = nodeFeatureOffset + numSplits * statSize;
    for (i = 0; i < statSize; i++) {
        rightImpurityCalculator[i] = allStats[lastOffset + i] - allStats[leftOffset + i];
        rightCount += rightImpurityCalculator[i];
    }
    double leftStats = calculateImpurity(impurity, allStats, statSize, leftOffset);
    double rightStats = calculateImpurity(impurity, rightImpurityCalculator, statSize, 0);
    double totalCount = leftCount + rightCount;
    if (leftCount < minInsPerNode || rightCount < minInsPerNode) {  // Invalid ImpurityStats
        ImpurityStats[0] = DBL_MIN;
        ImpurityStats[1] = impurityResult;
        if (ImpurityStats[2 + statSize * 3] > 0)
            memcpy(ImpurityStats + 2, parentImpurityCalculator, statSize * SZD);
        ImpurityStats[2 + statSize * 3] = -1;
        return ImpurityStats;
    }
    double leftWeight = leftCount / totalCount;
    double rightWeight = rightCount / totalCount;
    double gain = impurityResult - leftWeight * leftStats - rightWeight * rightStats;
    if (gain < minInfoGain) {
        ImpurityStats[0] = DBL_MIN;
        ImpurityStats[1] = impurityResult;
        if (ImpurityStats[2 + statSize * 3] > 0)
            memcpy(ImpurityStats + 2, parentImpurityCalculator, statSize * SZD);
        ImpurityStats[2 + statSize * 3] = -1;
        return ImpurityStats;
    }
    ImpurityStats[0] = gain;
    ImpurityStats[1] = impurityResult;
    memcpy(ImpurityStats + 2, parentImpurityCalculator, statSize * SZD);
    memcpy(ImpurityStats + 2 + statSize, allStats + leftOffset, statSize * SZD);
    memcpy(ImpurityStats + 2 + 2 * statSize, rightImpurityCalculator, statSize * SZD);
    ImpurityStats[2 + statSize * 3] = 1;
    return ImpurityStats;
}

/**
 * Result to Scala: retSplitAndStats
 * bestSplitIndex: Double
 * bestGain: Double
 * impurity: Double
 * impurityCalculator: Array[Double](statSize)
 * leftImpurityCalculator: Array[Double](statSize)
 * rightImpurityCalculator: Array[Double](statSize)
 * allStats: Array[Double](allStatsSize)
 * valid: double = 1 (-1 rep. false, 1 rep. true)
 *
 */
double* binToBestSplit(double ImpurityStats[], double allStats[], int featureOffset[], int nfeatureOffset,
                       int numSplits, char impurity, int statSize, int featureIndexIdx, int minInsPerNode, double minInfoGain) {
    int i, j;
    int allStatsSize = featureOffset[nfeatureOffset - 1];

    int nodeFeatureOffset = featureOffset[featureIndexIdx];
    // Cumulative sum (scanLeft) of bin statistics.
    // Afterwards, binAggregates for a bin is the sum of aggregates for
    // that bin + all preceding bins.

    for (i = 0; i < numSplits; i++) {  // split Index
        int toOffset = nodeFeatureOffset + (i+1) * statSize;
        int fromOffset = nodeFeatureOffset + i * statSize;
        for (j = 0; j < statSize; j++) {
            allStats[toOffset + j] += allStats[fromOffset + j];
        }
    }
    // get max gain ImpurityStats
    double bestImpurityStats[3 + statSize * 3];
    memset(bestImpurityStats, -1, sizeof(bestImpurityStats));
    double bestGain = -DBL_MAX;
    double bestSplitIndex = 0;
    for (i = 0; i < numSplits; i++) {
        ImpurityStats = calcGainAndImpurityStats(impurity, ImpurityStats, statSize, numSplits, allStats, allStatsSize,
                                    nodeFeatureOffset, nodeFeatureOffset + i * statSize, minInsPerNode, minInfoGain);
        if (ImpurityStats == NULL) continue;
        if (i == 0 || ImpurityStats[0] > bestGain) {  // is first or bigger than best
            memcpy(bestImpurityStats, ImpurityStats, (3 + statSize * 3) * SZD);
            bestGain = ImpurityStats[0];
            bestSplitIndex = i;
        }
    }
    double *retSplitAndStats = (double *) malloc((4 + statSize * 3) * SZD);
    retSplitAndStats[0] = bestSplitIndex;
    retSplitAndStats[1] = bestGain;
    retSplitAndStats[2] = bestImpurityStats[1];
    for (i = 0; i < statSize; i++) retSplitAndStats[3 + i] = bestImpurityStats[2 + i];
    for (i = 0; i < statSize; i++) retSplitAndStats[3 + i + statSize] = bestImpurityStats[2 + i + statSize];
    for (i = 0; i < statSize; i++) retSplitAndStats[3 + i + 2 * statSize] = bestImpurityStats[2 + i + 2 * statSize];
    retSplitAndStats[3 + statSize * 3] = bestImpurityStats[2 + statSize * 3];
    return retSplitAndStats;
}

/**
 * Result to Scala: retSplitAndStats
 * bestSplitIndex: Double
 * bestGain: Double
 * impurity: Double
 * impurityCalculator: Array[Double](statSize)
 * leftImpurityCalculator: Array[Double](statSize)
 * rightImpurityCalculator: Array[Double](statSize)
 * allStats: Array[Double](allStatsSize)
 * valid: double = 1 (-1 rep. false, 1 rep. true)
 *
 */

/**
 * ImpurityStats Representation:
 * gain: Double
 * impurity: Double
 * impurityCalculator: Array[Double]
 * leftImpurityCalculator: Array[Double]
 * rightImpurityCalculator: Array[Double]
 * valid: double = 1 (-1 rep. false, 1 rep. true)
 *
 * thus, [gain, impurity, {_, _, _}, {_, _, _}, {_, _, _}, valid]
 */
