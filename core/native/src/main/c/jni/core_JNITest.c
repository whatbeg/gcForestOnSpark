#include <stdio.h>
#include "core_JNITest.h"

JNIEXPORT jdouble JNICALL Java_core_JNITest_sum(JNIEnv *env, jobject thisObj, jdoubleArray arr, jint n) {
    jdouble *inArr = (*env)->GetDoubleArrayElements(env, arr, NULL);
    if (NULL == inArr) return (jdouble)0;
    jdouble sumi = 0;
    int i;
    for (i = 0; i < n; i++)
        sumi += inArr[i];
    return sumi;
}

JNIEXPORT jdouble JNICALL Java_core_JNITest_calc
  (JNIEnv *env, jobject thisObj, jdoubleArray arr, jint n) {
    jdouble *inArr = (*env)->GetDoubleArrayElements(env, arr, NULL);
    if (NULL == inArr) return (jdouble)0;
    jdouble mess = 1.0;
    int i;
    for (i = 0; i < n; i++) {
        mess = mess + (2 * inArr[i] + 3 * inArr[i] + 4 * inArr[i]) / (5 * inArr[i] + 1.0 + 2 * (jdouble)(inArr[i] - 1 + 2.4 + 1.0 * 3.0 / 1.0));
        mess = mess + mess - mess * mess + mess / mess + mess * mess;
        mess *= (jdouble)1.0;
        if (mess > 10000000000.0)
            mess /= (jdouble)10000000000.0;
    }
    (*env)->ReleaseDoubleArrayElements(env, arr, inArr, 0);
    return mess;
}

JNIEXPORT jdouble JNICALL Java_core_JNITest_calcIntensive
  (JNIEnv *env, jobject thisObj, jdouble a, jint n) {
    jdouble res = 1.0;
    int i;
    for (i = 1; i < n; i++) {
        res = res * 1.0 * 2.0 * 3.0 * 4.0 / 6.0 / 2.0 / 2.0 + res * 1.0 * 2.0 * 3.0 * 4.0 / 6.0 / 2.0 / 2.0;
        res = res * 10.0 * 20.0 * 30.0 * 40.0 / 60.0 / 20.0 / 20.0 + res * 10.0 * 20.0 * 30.0 * 40.0 / 60.0 / 20.0 / 20.0;
        res = res * (jdouble)i - res * (jdouble)i + res;
        res = res * 111111.0 / 111111.0;
        res = res / 10000.0 * 10000.0;
        res = res * 107070707 / 107070707;
    }
    return res;
}