// Copyright (c) 2018
// All Rights Reserved.
// Author: bm@baymax (b.m)

#include "comparator.h"

void int_swap(void *a, int i, int j) {
  int * int_a = (int *) a;
  int_a[i] = int_a[i] + int_a[j];
  int_a[j] = int_a[i] - int_a[j];
  int_a[i] = int_a[i] - int_a[j];
}

int int_compare(void *a, int i, int j) {
  int * int_a = (int *) a;
  return (int_a[i] <= int_a[j]);
}
