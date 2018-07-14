// Copyright (c) 2018
// All Rights Reserved.
// Author: bm@baymax (b.m)

#include <stdio.h>

void insertion_sort(void *, int,
                    int compare(void *, int, int),
                    void swap(void *, int, int));
int int_compare(void *, int, int);
void int_swap(void *, int, int);

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

void int_swap(void *a, int i, int j) {
  int * int_a = (int *) a;
  int x = int_a[i];
  int y = int_a[j];
  int_a[i] = y;
  int_a[j] = x;
}

int int_compare(void *a, int i, int j) {
  int * int_a = (int *) a;
  return (int_a[i] <= int_a[j]);
}

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
