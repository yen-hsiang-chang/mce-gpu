#pragma once
#include <cstdio>
#include <cuda_runtime_api.h>

#define CUDA_RUNTIME(ans)                 \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(1);
  }
}

#define execKernel(kernel, gridSize, blockSize, deviceId, verbose, ...) \
  {                                                                     \
    dim3 grid(gridSize);                                                \
    dim3 block(blockSize);                                              \
    CUDA_RUNTIME(cudaSetDevice(deviceId));                              \
    kernel<<<grid, block>>>(__VA_ARGS__);                               \
    CUDA_RUNTIME(cudaDeviceSynchronize());                              \
  }

#define execKernelAsync(kernel, gridSize, blockSize, deviceId, streamId, verbose, ...) \
  {                                                                                    \
    dim3 grid(gridSize);                                                               \
    dim3 block(blockSize);                                                             \
    CUDA_RUNTIME(cudaSetDevice(deviceId));                                             \
    kernel<<<grid, block, 0, streamId>>>(__VA_ARGS__);                                 \
  }

struct CUDAContext
{
  uint32_t max_threads_per_SM;
  uint32_t num_SMs;
  uint32_t shared_mem_size_per_block;
  uint32_t shared_mem_size_per_sm;

  CUDAContext()
  {
    /*get the maximal number of threads in an SM*/
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); /*currently 0th device*/
    max_threads_per_SM = prop.maxThreadsPerMultiProcessor;
    shared_mem_size_per_block = prop.sharedMemPerBlock;
    shared_mem_size_per_sm = prop.sharedMemPerMultiprocessor;
    num_SMs = prop.multiProcessorCount;
  }

  uint32_t GetConCBlocks(uint32_t block_size)
  {
    auto conc_blocks_per_SM = max_threads_per_SM / block_size; /*assume regs are not limited*/
    return conc_blocks_per_SM;
  }
};

__device__ __inline__ uint32_t __mysmid()
{
  unsigned int r;
  asm("mov.u32 %0, %%smid;"
      : "=r"(r));
  return r;
}

static __inline__ __device__ bool atomicCASBool(bool *address, bool compare, bool val)
{
  unsigned long long addr = (unsigned long long)address;
  unsigned pos = addr & 3;             // byte position within the int
  int *int_addr = (int *)(addr - pos); // int-aligned address
  int old = *int_addr, assumed, ival;

  do
  {
    assumed = old;
    if (val)
      ival = old | (1 << (8 * pos));
    else
      ival = old & (~((0xFFU) << (8 * pos)));
    old = atomicCAS(int_addr, assumed, ival);
  } while (assumed != old);

  return (bool)(old & ((0xFFU) << (8 * pos)));
}

template <typename T>
T getVal(T *arr, T index, AllocationTypeEnum at)
{
  if (at == AllocationTypeEnum::unified)
    return (arr[index]);

  T val = 0;
  CUDA_RUNTIME(cudaMemcpy(&val, &(arr[index]), sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
  return val;
}