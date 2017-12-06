/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>

typedef struct UserStruct {
    long long id;
    wchar_t* name;
    int age;
}UserStruct;

int add(int a, int b);

int add(int a, int b) {
    int c = a + b;
    return c;
}

double sum(double arr[], int n);

double sum(double arr[], int n) {
    double sum = 0;
    int i = 0;
    for (i; i < n; i++)
        sum += arr[i];
    return sum;
}

double calc(double arr[], int n);

double calc(double arr[], int n) {
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

double calcIntensive(double a, int n);

double calcIntensive(double a, int n) {
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



//#define MYLIBAPI extern "C" __declspec(dllexport)

void sayUser(UserStruct*);

void sayUser(UserStruct* pUserStruct) {
    printf("%lld\n", pUserStruct -> id);
    printf("%ls\n", pUserStruct -> name);
    printf("%d\n", pUserStruct -> age);
}

double* getSomeContentsForPointer(long long size);

double* getSomeContentsForPointer(long long size) {
    double a[20];
    memset(a, 0, sizeof(a));
    a[2] = 1;
    a[5] = 2;
    a[10] = 3;
    a[15] = 5;
    printf("%lld\n", size);
    double *p = (double *) malloc(size * sizeof(double));
    int i;
    for (i = 0; i < size; i++) p[i] = 0;
    p[0] = 11;
    p[1] = 12;
    return p;
}