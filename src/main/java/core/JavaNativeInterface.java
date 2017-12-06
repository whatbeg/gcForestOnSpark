/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package core;

public class JavaNativeInterface {
    static {
        System.loadLibrary("binToBestSplit");
    }

    public native double[] binToBestSplit(double[] ImpurityStats,
                                          double[] allStats,
                                          int[] featureOffset,
                                          int nfeatureOffset,
                                          int numSplits,
                                          char impurity,
                                          int statSize,
                                          int featureIndexIdx,
                                          int minInsPerNode,
                                          double minInfoGain);
    public static void main(String[] args) {

    }
}
