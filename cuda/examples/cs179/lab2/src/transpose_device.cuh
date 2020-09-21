#pragma once

enum TransposeImplementation { NAIVE, SHMEM, OPTIMAL };

void cudaTranspose(const float *d_input, float *d_output, int n, TransposeImplementation type);
