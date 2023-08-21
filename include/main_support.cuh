#pragma once

#include <cuda_runtime_api.h>
#include "csrcoo.cuh"

template <typename T>
__global__ void set_priority(graph::COOCSRGraph_d<T> g, T m, T *priority, T *tmp_block, T *split_ptr)
{
  auto tx = threadIdx.x, bx = blockIdx.x, ptx = tx + bx * blockDim.x;

  if (ptx < m)
  {
    const T src = g.rowInd[ptx], dst = g.colInd[ptx];
    bool keep = priority[src] == priority[dst] ? src < dst : priority[src] < priority[dst];
    if (!keep)
    {
      atomicAdd(tmp_block + bx, 1);
      atomicAdd(split_ptr + src, 1);
    }
  }
}

template <typename T>
__global__ void split_pointer(graph::COOCSRGraph_d<T> g, T *split_ptr)
{
  auto tx = threadIdx.x, bx = blockIdx.x, ptx = tx + bx * blockDim.x;
  if (ptx <= g.numNodes)
    split_ptr[ptx] += g.rowPtr[ptx];
}

template <typename T, size_t BLOCK_DIM_X>
__global__ void split_data(graph::COOCSRGraph_d<T> g, T m, T *priority, T *tmp_block, T *split_ptr, T *gsplit_ptr, T *gsplit_col)
{
  __shared__ short prefix[BLOCK_DIM_X];
  auto tx = threadIdx.x, bx = blockIdx.x, ptx = tx + bx * blockDim.x;
  bool keep;
  T src = 0, dst = 0;

  prefix[tx] = 0;
  __syncthreads();
  if (ptx < m)
  {
    src = g.rowInd[ptx];
    dst = g.colInd[ptx];
    keep = priority[src] == priority[dst] ? src < dst : priority[src] < priority[dst];
    prefix[tx] = 1 - keep;
  }
  __syncthreads();
  {
    int stride = 1;
    while (stride < BLOCK_DIM_X)
    {
      auto idx = (tx + 1) * stride * 2 - 1;
      if (idx < BLOCK_DIM_X && idx >= stride)
        prefix[idx] += prefix[idx - stride];
      stride <<= 1;
      __syncthreads();
    }
  }
  {
    int stride = BLOCK_DIM_X >> 2;
    while (stride)
    {
      auto idx = (tx + 1) * stride * 2 - 1;
      if ((idx + stride) < BLOCK_DIM_X)
        prefix[idx + stride] += prefix[idx];
      stride >>= 1;
      __syncthreads();
    }
  }
  if (ptx < m)
  {
    if (!keep)
    {
      T idx = tmp_block[bx] + prefix[tx] - 1 - split_ptr[src];
      gsplit_col[g.rowPtr[src] + idx] = dst;
    }
    else
    {
      T idx = ptx - tmp_block[bx] - prefix[tx] - (g.rowPtr[src] - split_ptr[src]);
      gsplit_col[gsplit_ptr[src] + idx] = dst;
    }
  }
}
