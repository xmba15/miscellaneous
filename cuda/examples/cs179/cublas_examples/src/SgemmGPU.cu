/**
 * @file    SgemmGPU.cu
 *
 * @author  btran
 *
 */

#include "SgemmGPU.cuh"

#include <cublas_v2.h>
#include <thrust/device_vector.h>

namespace cuda
{
void sgemmGPU(int n, float alpha, const float* A, const float* B, float beta, float* C)
{
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);

    int n2 = n * n;

    thrust::device_vector<float> dA(A, A + n2);
    thrust::device_vector<float> dB(B, B + n2);
    thrust::device_vector<float> dC(C, C + n2);

    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, thrust::raw_pointer_cast(dA.data()), n,
                         thrust::raw_pointer_cast(dB.data()), n, &beta, thrust::raw_pointer_cast(dC.data()), n);

    thrust::copy(dC.begin(), dC.end(), C);

    status = cublasDestroy(handle);
}
}  // namespace cuda
