/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package core;

public class JNITest {
    static {
        System.loadLibrary("core_JNITest");
    }

    public native double sum(double[] arr, int n);
    public native double calc(double[] arr, int n);
    public native double calcIntensive(double a, int n);

    public double sumJava(double[] arr, int n) {
        double sum = 0.0;
        int i;
        for (i = 0; i < n; i++) {
            sum += arr[i];
        }
        return sum;
    }

    public double calcJava(double[] arr, int n) {
        double mess = 1.0;
        int i;
        for (i = 0; i < n; i++) {
            mess = mess + (2 * arr[i] + 3 * arr[i] + 4 * arr[i]) / (5 * arr[i] + 1.0 + 2 * (arr[i] - 1 + 2.4 + 1.0 * 3.0 / 1.0));
            mess = mess + mess - mess * mess + mess / mess + mess * mess;
            mess *= 1.0;
            if (mess > 10000000000.0)
                mess /= 10000000000.0;
        }
        return mess;
    }

    public double calcIntensiveJava(double a, int n) {
        double res = 1.0;
        int i;
        for (i = 1; i < n; i++) {
            res = res * 1.0 * 2.0 * 3.0 * 4.0 / 6.0 / 2.0 / 2.0 + res * 1.0 * 2.0 * 3.0 * 4.0 / 6.0 / 2.0 / 2.0;
            res = res * 10.0 * 20.0 * 30.0 * 40.0 / 60.0 / 20.0 / 20.0 + res * 10.0 * 20.0 * 30.0 * 40.0 / 60.0 / 20.0 / 20.0;
            res = res * i - res * i + res;
            res = res * 111111.0 / 111111.0;
            res = res / 10000.0 * 10000.0;
            res = res * 107070707 / 107070707;
        }
        return res;
    }

    public static void main(String[] args) {
        double[] s = {0, 1, 2, 3};
        double res = new JNITest().sum(s, 4);
        System.out.println(res);
    }
}
