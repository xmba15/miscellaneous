// Copyright (c) 2018
// All Rights Reserved.
// Author: bm@baymax (b.m)

#include <stdio.h>
#include "comparator.h"
#include "insertion_sort.h"

void printa(int *a, int n) {
  for (int i = 0; i < n; i++)
    printf("%d ", a[i]);
  putchar('\n');
}

int main(int argc, char *argv[]) {
  int a[] = {1, 3, 4, 5, 8, 20, 6, 100, 5};
  int n = sizeof(a) / sizeof(a[0]);
  printa(a, n);
  insertion_sort(a, n, int_compare, int_swap);
  printa(a, n);

  return 0;
}

