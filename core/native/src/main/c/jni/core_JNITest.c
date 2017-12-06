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