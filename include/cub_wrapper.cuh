#pragma once

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include "defs.h"
#include "utils.cuh"

template <typename InputType, typename OutputType, typename CountType, typename FlagIterator>
uint32_t CUBSelect(
    InputType input, OutputType output,
    FlagIterator flags,
    const CountType countInput,
    int devId)
{
  CUDA_RUNTIME(cudaSetDevice(devId));
  uint32_t *countOutput = nullptr;
  CUDA_RUNTIME(cudaMallocManaged(&countOutput, sizeof(uint32_t)));

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, input, flags, output, countOutput, countInput);
  CUDA_RUNTIME(cudaDeviceSynchronize());
  CUDA_RUNTIME(cudaMallocManaged(&d_temp_storage, temp_storage_bytes));
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, input, flags, output, countOutput, countInput);
  CUDA_RUNTIME(cudaDeviceSynchronize());
  uint32_t res = *countOutput;
  CUDA_RUNTIME(cudaFree(d_temp_storage));
  CUDA_RUNTIME(cudaFree(countOutput));
  return res;
}

template <typename InputType, typename OutputType>
OutputType CUBScanExclusive(
    InputType *input, OutputType *output,
    const int count, int devId,
    cudaStream_t stream = 0, AllocationTypeEnum at = unified)
{
  CUDA_RUNTIME(cudaSetDevice(devId));

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  /*record the last input item in case it is an in-place scan*/
  auto last_input = getVal<InputType>(input, count - 1, at); // input[count - 1];
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input, output, count);
  CUDA_RUNTIME(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input, output, count);
  CUDA_RUNTIME(cudaFree(d_temp_storage));

  return getVal<OutputType>(output, count - 1, at) + (OutputType)last_input;
}
