/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package core;

public class JNITest {
    static {
        System.loadLibrary("core_JNITest");
    }

    public native double sum(double[] arr, int n);

    public double sumJava(double[] arr, int n) {
        double sum = 0.0;
        int i;
        for (i = 0; i < n; i++) {
            sum += arr[i];
        }
        return sum;
    }

    public static void main(String[] args) {
        double[] s = {0, 1, 2, 3};
        double res = new JNITest().sum(s, 4);
        System.out.println(res);
    }
}
