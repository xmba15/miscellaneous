/**
 * @file    SgemmGPU.cuh
 *
 * @author  btran
 *
 */

#pragma once

namespace cuda
{
void sgemmGPU(int n, float alpha, const float* A, const float* B, float beta, float* C);
}  // namespace cuda
