/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
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