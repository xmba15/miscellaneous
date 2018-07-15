// Copyright (c) 2018
// All Rights Reserved.
// Author: bm@baymax (b.m)

#include "insertion_sort.h"

void insertion_sort(void *arr, int n,
                    int compare(void *, int, int),
                    void swap(void *, int, int)) {
  int i, j;
  for (i = 0; i < n; i++)
    for (j = i + 1; j < n; j++) {
      if (!compare(arr, i, j))
        swap(arr, i, j);
    }
}
