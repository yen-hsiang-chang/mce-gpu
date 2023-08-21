#pragma once

#include "../include/csrcoo.cuh"
#include "../include/defs.h"
#include "../include/queue.cuh"
#include "parameter.cuh"

template <typename T>
struct GLOBAL_HANDLE {
  graph::COOCSRGraph_d<T> gsplit;
  T iteration_limit;
  T *level_candidate;
  uint64 *mce_counter;
  T *encoded_induced_subgraph;

  T *P;
  T *Xp;
  T *level_pointer;
  T *Xx;
  T *Xx_aux;
  T *tri_list;
  cuda::atomic<uint32_t, cuda::thread_scope_device> *work_ready;
  uint32_t *global_message;
  volatile T *work_stealing;
  T stride;
};

template<typename T>
struct LOCAL_HANDLE {
  uint numPartitions, wx, lx, partMask;
  T maskBlock, maskIndex, newIndex, sameBlockMask;
};

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
struct SHARED_HANDLE {
  T level_pivot;
  bool path_more_explore, path_eliminated;
  T l, base_l, sm_block_id, root_sm_block_id;
  T maxIntersection;
  uint32_t state, worker_pos, shared_other_sm_block_id;
  T src, srcStart, srcLen, curSZ, usrcStart, usrcLen, cnodeStart, cnodeLen, cnode;
  T src2, src2Start, src2Len;
  bool x_empty;

  T num_divs_local;
  T *encode;
  T *pl, *cl, *xl;
  T *tri, scounter;
  T *level_pointer_index;

  T maxCount[BLOCK_DIM_X / CPARTSIZE], maxIndex[BLOCK_DIM_X / CPARTSIZE];
  T lastMask_i, lastMask_ii;
  uint64 count;
  T *Xx_shared, *Xx_aux_shared, Xx_sz[MAX_TRAVERSE_LEVELS + 1], Xx_aux_sz;
  T to_cl[MAX_DEGEN / 32], to_xl[MAX_DEGEN / 32];
  bool fork;
  T i;
  T frontier;
};

namespace graph
{
  template <typename T>
  __host__ __device__ T binary_search(const T *arr, const T lt, const T rt, const T searchVal, bool &found)
  {
    T left = lt, right = rt;
    found = false;
    while (left < right)
    {
      const T mid = (left + right) / 2;
      T val = arr[mid];
      if (val == searchVal)
      {
        found = true;
        return mid;
      }
      bool pred = val < searchVal;
      if (pred)
      {
        left = mid + 1;
      }
      else
      {
        right = mid;
      }
    }
    return left;
  }

  template <typename T, uint CPARTSIZE = 32>
  __device__ __forceinline__ uint64 warp_sorted_count_and_encode_full(
      const T *const A, const size_t aSz, T *B,
      T bSz, T j, T num_divs_local, T *encode)
  {
    const int warpIdx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const int laneIdx = threadIdx.x % CPARTSIZE; // which thread in warp
    // cover entirety of A with warp
    for (T i = laneIdx; i < aSz; i += CPARTSIZE)
    {
      const T searchVal = A[i];
      bool found = false;
      const T lb = graph::binary_search<T>(B, 0, bSz, searchVal, found);
      if (found)
      {
        //////////////////////////////Device function ///////////////////////
        T chunk_index = i / 32; // 32 here is the division size of the encode
        T inChunkIndex = i % 32;
        atomicOr(&encode[j * num_divs_local + chunk_index], 1 << inChunkIndex);

        T chunk_index1 = j / 32; // 32 here is the division size of the encode
        T inChunkIndex1 = j % 32;
        atomicOr(&encode[i * num_divs_local + chunk_index1], 1 << inChunkIndex1);
        /////////////////////////////////////////////////////////////////////
      }
    }
    return 0;
  }

  template <typename T, uint CPARTSIZE = 32>
  __device__ __forceinline__ uint64 warp_sorted_count_and_encode_full_mclique(
      const T *const A, const size_t aSz, T *B,
      T bSz, T j, T num_divs_local, T *encode, T base)
  {
    const int warpIdx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const int laneIdx = threadIdx.x % CPARTSIZE; // which thread in warp
    // cover entirety of A with warp
    for (T i = laneIdx; i < aSz; i += CPARTSIZE)
    {
      const T searchVal = A[i];
      bool found = false;
      const T lb = graph::binary_search<T>(B, 0, bSz, searchVal, found);
      if (found)
      {
        //////////////////////////////Device function ///////////////////////
        T chunk_index = i / 32; // 32 here is the division size of the encode
        T inChunkIndex = i % 32;
        atomicOr(&encode[(j >= base ? j - base : j + aSz) * num_divs_local + chunk_index], 1 << inChunkIndex);

        if (j >= base)
        {
          T chunk_index1 = (j - base) / 32; // 32 here is the division size of the encode
          T inChunkIndex1 = (j - base) % 32;
          atomicOr(&encode[i * num_divs_local + chunk_index1], 1 << inChunkIndex1);
        }
        /////////////////////////////////////////////////////////////////////
      }
    }
    return 0;
  }

  template <size_t BLOCK_DIM_X, typename T>
  __device__ __forceinline__ uint64 block_count_and_set_tri(
      const T *const A, const size_t aSz,
      const T *const B, const size_t bSz,
      T *tri, T *counter)
  {
    // cover entirety of A with block
    for (size_t i = threadIdx.x; i < aSz; i += BLOCK_DIM_X)
    {
      // one element of A per thread, just search for A into B
      const T searchVal = A[i];
      bool found = false;
      binary_search<T>(B, 0, bSz, searchVal, found);
      if (found)
      {
        T old = atomicAdd(counter, 1);
        tri[old] = searchVal;
      }
    }
    return 0;
  }
} // namespace graph

template <typename T, uint CPARTSIZE>
__device__ __forceinline__ void reduce_part(T partMask, uint64 &warpCount)
{
  for (int i = CPARTSIZE / 2; i >= 1; i /= 2)
    warpCount += __shfl_down_sync(partMask, warpCount, i);
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void encode_clear(
    LOCAL_HANDLE<T> &lh, SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> & sh, const T &lim)
{
  for (T j = lh.wx; j < lim; j += lh.numPartitions)
    for (T k = lh.lx; k < sh.num_divs_local; k += CPARTSIZE)
      sh.encode[j * sh.num_divs_local + k] = 0x00;
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void finalize_pivot(
    LOCAL_HANDLE<T> &lh, SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh, const T &lim)
{
  if (lh.lx == 0 && sh.maxCount[lh.wx] != lim + 1)
    atomicMax(&(sh.maxIntersection), sh.maxCount[lh.wx]);
  __syncthreads();

  if (lh.lx == 0 && sh.maxIntersection == sh.maxCount[lh.wx])
    sh.level_pivot = sh.maxIndex[lh.wx];
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void apply_pivot_to_first_level(
    SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh)
{
  for (T j = threadIdx.x; j < sh.num_divs_local; j += blockDim.x)
  {
    const T m = (j == sh.lastMask_i) ? sh.lastMask_ii : 0xFFFFFFFF;
    sh.pl[j] = ~(sh.encode[(sh.level_pivot)*sh.num_divs_local + j]) & m;
    sh.cl[j] = m;
    sh.xl[j] = 0;
  }
  __syncthreads();
}

template <bool WL, typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void apply_pivot_to_next_level(
    SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh, const T &level)
{
  for (T j = threadIdx.x; j < sh.num_divs_local; j += blockDim.x)
  {
    T m = (j == sh.lastMask_i) ? sh.lastMask_ii : 0xFFFFFFFF;
    T mask = ~(sh.encode[sh.level_pivot * sh.num_divs_local + j]) & sh.to_cl[j] & m;
    sh.pl[level * sh.num_divs_local + j] = mask;
    if (WL)
    {
      T idx = 0;
      while ((idx = __ffs(mask)))
      {
        --idx;
        sh.Xx_aux_shared[atomicAdd(&sh.Xx_aux_sz, 1)] = j * 32 + idx;
        mask ^= 1u << idx;
      }
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void wait_for_donor(
    cuda::atomic<uint32_t, cuda::thread_scope_device> &work_ready, uint32_t &shared_state,
    queue_callee(queue, tickets, head, tail))
{
  uint32_t ns = 8;
  do
  {
    if (work_ready.load(cuda::memory_order_relaxed))
    {
      if (work_ready.load(cuda::memory_order_acquire))
      {
        shared_state = 2;
        work_ready.store(0, cuda::memory_order_relaxed);
        break;
      }
    }
    else if (queue_full(queue, tickets, head, tail, CB))
    {
      shared_state = 100;
      break;
    }
  } while (ns = my_sleep(ns));
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void Xx_compaction_for_IP(
    GLOBAL_HANDLE<T> &gh,
    SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh,
    const T &level)
{
  const T lim = sh.Xx_sz[level];
  for (T j = threadIdx.x; j < lim; j += blockDim.x)
  {
    bool found = false;
    T cur = sh.Xx_shared[j];
    graph::binary_search(gh.gsplit.colInd + sh.cnodeStart, 0u, sh.cnodeLen, gh.gsplit.colInd[sh.usrcStart + cur], found);
    if (found)
    {
      sh.Xx_aux_shared[atomicAdd(&sh.Xx_sz[level + 1], 1)] = cur;
    }
    else
    {
      sh.Xx_aux_shared[lim - atomicAdd(&sh.Xx_aux_sz, 1) - 1] = cur;
    }
  }
  __syncthreads();
  for (T j = threadIdx.x; j < lim; j += blockDim.x)
  {
    sh.Xx_shared[j] = sh.Xx_aux_shared[j];
  }
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void Xx_compaction_for_IPX(
    LOCAL_HANDLE<T> &lh,
    SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh,
    const T &level)
{
  const T lim = sh.Xx_sz[level];
  for (T j = threadIdx.x; j < lim; j += blockDim.x)
  {
    T cur = sh.Xx_shared[j];
    if ((sh.encode[cur * sh.num_divs_local + (lh.newIndex >> 5)] >> (lh.newIndex & 0x1F)) & 1)
    {
      sh.Xx_aux_shared[atomicAdd(&sh.Xx_sz[level + 1], 1)] = cur;
    }
    else
    {
      sh.Xx_aux_shared[lim - atomicAdd(&sh.Xx_aux_sz, 1) - 1] = cur;
    }
  }
  __syncthreads();
  for (T j = threadIdx.x; j < lim; j += blockDim.x)
  {
    sh.Xx_shared[j] = sh.Xx_aux_shared[j];
  }
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void populate_xl_and_cl(
    LOCAL_HANDLE<T> &lh, 
    SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh,
    const T &level)
{
  for (T k = threadIdx.x; k < sh.num_divs_local; k += blockDim.x)
  {
    sh.to_xl[k] = sh.xl[sh.num_divs_local * level + k] | ((lh.maskBlock < k) ? sh.pl[sh.num_divs_local * level + k] : ((lh.maskBlock > k) ? 0 : ~lh.sameBlockMask));
    sh.to_xl[k] &= sh.encode[lh.newIndex * sh.num_divs_local + k];
    sh.xl[sh.num_divs_local * (level + 1) + k] = sh.to_xl[k];
    sh.to_cl[k] = sh.cl[sh.num_divs_local * level + k] & sh.encode[lh.newIndex * sh.num_divs_local + k];
    sh.to_cl[k] &= ((lh.maskBlock < k) ? ~sh.pl[sh.num_divs_local * level + k] : ((lh.maskBlock > k) ? 0xFFFFFFFF : lh.sameBlockMask));
    sh.cl[sh.num_divs_local * (level + 1) + k] = sh.to_cl[k];
  }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void select_pivot_from_P(
    LOCAL_HANDLE<T> &lh, SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh, const T &lim)
{
  for (T j = lh.wx; j < lim; j += lh.numPartitions)
  {
    uint64 warpCount = 0;
    for (T k = lh.lx; k < sh.num_divs_local; k += CPARTSIZE)
    {
      warpCount += __popc(sh.encode[j * sh.num_divs_local + k]);
    }
    reduce_part<T, CPARTSIZE>(lh.partMask, warpCount);

    if (lh.lx == 0 && sh.maxCount[lh.wx] == lim + 1)
    {
      sh.path_more_explore = true; // shared, unsafe, but okay
      sh.maxCount[lh.wx] = warpCount;
      sh.maxIndex[lh.wx] = j;
    }
    else if (lh.lx == 0 && sh.maxCount[lh.wx] < warpCount)
    {
      sh.maxCount[lh.wx] = warpCount;
      sh.maxIndex[lh.wx] = j;
    }
  }
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void select_pivot_from_P_and_Xp(
    LOCAL_HANDLE<T> &lh, SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh, const T &lim)
{
  for (T j = lh.wx; j < lim; j += lh.numPartitions)
  {
    const T bi = j >> 5, ii = j & 0x1F;
    if ((sh.to_cl[bi] & (1 << ii)) != 0 || (sh.to_xl[bi] & (1 << ii)) != 0)
    {
      uint64 warpCount = 0;
      for (T k = lh.lx; k < sh.num_divs_local; k += CPARTSIZE)
      {
        warpCount += __popc(sh.to_cl[k] & sh.encode[j * sh.num_divs_local + k]);
      }
      reduce_part<T, CPARTSIZE>(lh.partMask, warpCount);

      /* A pivot from X removes all vertices from P */
      if (lh.lx == 0 && sh.curSZ == warpCount)
      {
        sh.path_eliminated = true;
      }

      if (lh.lx == 0 && sh.maxCount[lh.wx] == lim + 1)
      {
        sh.path_more_explore = true; // shared, unsafe, but okay
        sh.maxCount[lh.wx] = warpCount;
        sh.maxIndex[lh.wx] = j;
      }
      else if (lh.lx == 0 && sh.maxCount[lh.wx] < warpCount)
      {
        sh.maxCount[lh.wx] = warpCount;
        sh.maxIndex[lh.wx] = j;
      }
    }
  }
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void select_pivot_from_Xx_first_level(
    LOCAL_HANDLE<T> &lh, SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh,
    const T &Xx_sz, const T &lim)
{
  for (T j = lh.wx; j < Xx_sz; j += lh.numPartitions)
  {
    uint64 warpCount = 0;
    T cur = sh.Xx_shared[j];
    for (T k = lh.lx; k < sh.num_divs_local; k += CPARTSIZE)
    {
      warpCount += __popc(sh.encode[cur * sh.num_divs_local + k]);
    }
    reduce_part<T, CPARTSIZE>(lh.partMask, warpCount);

    if (lh.lx == 0 && lim == warpCount)
    {
      sh.path_eliminated = true;
    }

    if (lh.lx == 0 && sh.maxCount[lh.wx] == lim + 1)
    {
      sh.path_more_explore = true;
      sh.maxCount[lh.wx] = warpCount;
      sh.maxIndex[lh.wx] = cur;
    }
    else if (lh.lx == 0 && sh.maxCount[lh.wx] < warpCount)
    {
      sh.maxCount[lh.wx] = warpCount;
      sh.maxIndex[lh.wx] = cur;
    }
  }
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void select_pivot_from_Xx_not_first_level(
    LOCAL_HANDLE<T> &lh, SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh,
    const T &Xx_sz, const T &lim)
{
  for (T j = lh.wx; j < Xx_sz; j += lh.numPartitions)
  {
    uint64 warpCount = 0;
    T cur = sh.Xx_shared[j];
    for (T k = lh.lx; k < sh.num_divs_local; k += CPARTSIZE)
    {
      warpCount += __popc(sh.to_cl[k] & sh.encode[cur * sh.num_divs_local + k]);
    }
    reduce_part<T, CPARTSIZE>(lh.partMask, warpCount);

    if (lh.lx == 0 && sh.curSZ == warpCount)
    {
      sh.path_eliminated = true;
    }

    if (lh.lx == 0 && sh.maxCount[lh.wx] == lim + 1)
    {
      sh.path_more_explore = true;
      sh.maxCount[lh.wx] = warpCount;
      sh.maxIndex[lh.wx] = cur;
    }
    else if (lh.lx == 0 && sh.maxCount[lh.wx] < warpCount)
    {
      sh.maxCount[lh.wx] = warpCount;
      sh.maxIndex[lh.wx] = cur;
    }
  }
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void get_candidate_size(
    LOCAL_HANDLE<T> &lh, SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh)
{
  uint64 warpCount = 0;
  if (threadIdx.x == 0)
  {
    sh.curSZ = 0;
    sh.path_eliminated = false;
  }
  __syncthreads();

  for (T j = threadIdx.x; j < sh.num_divs_local; j += blockDim.x)
    warpCount += __popc(sh.to_cl[j]);
  reduce_part<T, CPARTSIZE>(lh.partMask, warpCount);

  if (lh.lx == 0 && threadIdx.x < sh.num_divs_local)
    atomicAdd(&sh.curSZ, warpCount);
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void init_max_count_and_index(
    LOCAL_HANDLE<T> &lh, SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh, const T &lim)
{
  if (lh.lx == 0)
  {
    sh.maxCount[lh.wx] = lim + 1;
    sh.maxIndex[lh.wx] = 0;
  }
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void test_maximality(
    SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh, const T &Xx_sz)
{
  if (Xx_sz != 0)
    return;
  if (threadIdx.x == 0)
    sh.x_empty = true;
  __syncthreads();

  for (T j = threadIdx.x; j < sh.num_divs_local; j += blockDim.x)
    if (sh.to_xl[j])
      sh.x_empty = false;
  __syncthreads();

  if (threadIdx.x == 0 && sh.x_empty)
    ++sh.count;
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ bool get_next_candidate(
    GLOBAL_HANDLE<T> &gh, LOCAL_HANDLE<T> &lh, 
    SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh,
    const T &lim, const T &level_pointer_index, T *cur_pl)
{
  if (level_pointer_index >= lim)
  {
    __syncthreads();
    if (threadIdx.x == 0)
      --sh.l;
    __syncthreads();
    return false;
  }
  lh.maskBlock = level_pointer_index >> 5;
  lh.maskIndex = ~((1 << (level_pointer_index & 0x1F)) - 1);
  lh.newIndex = __ffs(cur_pl[lh.maskBlock] & lh.maskIndex);
  while (lh.newIndex == 0)
  {
    lh.maskIndex = 0xFFFFFFFF;
    ++lh.maskBlock;
    if ((lh.maskBlock << 5) >= lim)
      break;
    lh.newIndex = __ffs(cur_pl[lh.maskBlock] & lh.maskIndex);
  }
  if ((lh.maskBlock << 5) >= lim)
  {
    __syncthreads();
    if (threadIdx.x == 0)
      --sh.l;
    __syncthreads();
    return false;
  }
  lh.newIndex = (lh.maskBlock << 5) + lh.newIndex - 1;
  lh.sameBlockMask = (~((1 << (lh.newIndex & 0x1F)) - 1)) | ~cur_pl[lh.maskBlock];
  __syncthreads();
  return true;
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void setup_stack_original_L1_IP(
    GLOBAL_HANDLE<T> &gh, SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh)
{
  if (threadIdx.x == 0)
  {
    sh.src = sh.i;
    sh.srcStart = gh.gsplit.splitPtr[sh.src];
    sh.srcLen = gh.gsplit.rowPtr[sh.src + 1] - sh.srcStart;
    sh.usrcStart = gh.gsplit.rowPtr[sh.src];
    sh.usrcLen = gh.gsplit.splitPtr[sh.src] - sh.usrcStart;

    size_t Xx_offset = sh.sm_block_id * MAXUNDEG;
    sh.Xx_shared = &gh.Xx[Xx_offset];
    sh.Xx_aux_shared = &gh.Xx_aux[Xx_offset];

    sh.num_divs_local = (sh.srcLen + 32 - 1) / 32;
    size_t encode_offset = sh.sm_block_id * (MAXDEG * NUMDIVS);
    sh.encode = &gh.encoded_induced_subgraph[encode_offset];

    size_t level_offset = sh.sm_block_id * NUMDIVS * (MAXDEG + 1);
    sh.cl = &gh.level_candidate[level_offset];
    sh.pl = &gh.P[level_offset];
    sh.xl = &gh.Xp[level_offset];

    size_t level_item_offset = sh.sm_block_id * (MAXDEG + 1);
    sh.level_pointer_index = &gh.level_pointer[level_item_offset];

    sh.level_pointer_index[0] = 0;
    sh.l = sh.base_l = 2;

    sh.level_pivot = 0xFFFFFFFF;

    sh.path_more_explore = false;
    sh.path_eliminated = false;
    sh.maxIntersection = 0;

    sh.lastMask_i = sh.srcLen >> 5;
    sh.lastMask_ii = (1 << (sh.srcLen & 0x1F)) - 1;
    sh.Xx_sz[0] = sh.usrcLen;
  }
  __syncthreads();

  for (T j = threadIdx.x; j < sh.usrcLen; j += blockDim.x)
    sh.Xx_shared[j] = j;
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void setup_stack_original_L1_IPX(
    GLOBAL_HANDLE<T> &gh, SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh)
{
  if (threadIdx.x == 0)
  {
    sh.src = sh.i;
    sh.srcStart = gh.gsplit.splitPtr[sh.src];
    sh.srcLen = gh.gsplit.rowPtr[sh.src + 1] - sh.srcStart;
    sh.usrcStart = gh.gsplit.rowPtr[sh.src];
    sh.usrcLen = gh.gsplit.splitPtr[sh.src] - sh.usrcStart;

    auto Xx_offset = sh.sm_block_id * MAXUNDEG;
    sh.Xx_shared = &gh.Xx[Xx_offset];
    sh.Xx_aux_shared = &gh.Xx_aux[Xx_offset];

    sh.num_divs_local = (sh.srcLen + 32 - 1) / 32;
    auto encode_offset = sh.sm_block_id * (MAXUNDEG * NUMDIVS);
    sh.encode = &gh.encoded_induced_subgraph[encode_offset];

    auto level_offset = sh.sm_block_id * NUMDIVS * (MAXDEG + 1);
    sh.cl = &gh.level_candidate[level_offset];
    sh.pl = &gh.P[level_offset];
    sh.xl = &gh.Xp[level_offset];

    auto level_item_offset = sh.sm_block_id * (MAXDEG + 1);
    sh.level_pointer_index = &gh.level_pointer[level_item_offset];

    sh.level_pointer_index[0] = 0;
    sh.l = sh.base_l = 2;

    sh.level_pivot = 0xFFFFFFFF;

    sh.path_more_explore = false;
    sh.path_eliminated = false;
    sh.maxIntersection = 0;

    sh.lastMask_i = sh.srcLen >> 5;
    sh.lastMask_ii = (1 << (sh.srcLen & 0x1F)) - 1;
    sh.Xx_sz[0] = sh.usrcLen;
  }
  __syncthreads();

  for (T j = threadIdx.x; j < sh.usrcLen; j += blockDim.x)
  {
    sh.Xx_shared[j] = j + sh.srcLen;
  }
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void setup_stack_original_L2_IP(
    GLOBAL_HANDLE<T> &gh, SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh)
{
  if (threadIdx.x == 0)
  {
    sh.src = gh.gsplit.rowInd[sh.i];
    sh.srcStart = gh.gsplit.splitPtr[sh.src];
    sh.srcLen = gh.gsplit.rowPtr[sh.src + 1] - sh.srcStart;
    sh.src2 = gh.gsplit.colInd[sh.i];
    sh.src2Start = gh.gsplit.splitPtr[sh.src2];
    sh.src2Len = gh.gsplit.rowPtr[sh.src2 + 1] - sh.src2Start;
    sh.usrcStart = gh.gsplit.rowPtr[sh.src];
    sh.usrcLen = gh.gsplit.splitPtr[sh.src] - sh.usrcStart;
    sh.cnode = sh.src2;
    sh.cnodeStart = gh.gsplit.rowPtr[sh.cnode];
    sh.cnodeLen = gh.gsplit.splitPtr[sh.cnode] - sh.cnodeStart;

    auto Xx_offset = sh.sm_block_id * MAXUNDEG;
    sh.Xx_shared = &gh.Xx[Xx_offset];
    sh.Xx_aux_shared = &gh.Xx_aux[Xx_offset];

    auto tri_offset = sh.sm_block_id * MAXDEG;
    sh.tri = &gh.tri_list[tri_offset];
    sh.scounter = 0;

    auto encode_offset = sh.sm_block_id * (MAXDEG * NUMDIVS);
    sh.encode = &gh.encoded_induced_subgraph[encode_offset];

    auto level_offset = sh.sm_block_id * NUMDIVS * MAXDEG;
    sh.cl = &gh.level_candidate[level_offset];
    sh.pl = &gh.P[level_offset];
    sh.xl = &gh.Xp[level_offset];

    auto level_item_offset = sh.sm_block_id * MAXDEG;
    sh.level_pointer_index = &gh.level_pointer[level_item_offset];

    sh.level_pointer_index[0] = 0;
    sh.l = sh.base_l = 3;

    sh.level_pivot = 0xFFFFFFFF;

    sh.path_more_explore = false;
    sh.path_eliminated = false;
    sh.maxIntersection = 0;
    sh.Xx_sz[0] = 0;
  }
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void setup_stack_original_L2_IPX(
    GLOBAL_HANDLE<T> &gh, SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh)
{
  if (threadIdx.x == 0)
  {
    sh.src = gh.gsplit.rowInd[sh.i];
    sh.srcStart = gh.gsplit.splitPtr[sh.src];
    sh.srcLen = gh.gsplit.rowPtr[sh.src + 1] - sh.srcStart;
    sh.src2 = gh.gsplit.colInd[sh.i];
    sh.src2Start = gh.gsplit.splitPtr[sh.src2];
    sh.src2Len = gh.gsplit.rowPtr[sh.src2 + 1] - sh.src2Start;
    sh.usrcStart = gh.gsplit.rowPtr[sh.src];
    sh.usrcLen = gh.gsplit.splitPtr[sh.src] - sh.usrcStart;
    sh.cnode = sh.src2;
    sh.cnodeStart = gh.gsplit.rowPtr[sh.cnode];
    sh.cnodeLen = gh.gsplit.splitPtr[sh.cnode] - sh.cnodeStart;

    auto Xx_offset = sh.sm_block_id * MAXUNDEG;
    sh.Xx_shared = &gh.Xx[Xx_offset];
    sh.Xx_aux_shared = &gh.Xx_aux[Xx_offset];

    auto tri_offset = sh.sm_block_id * MAXDEG;
    sh.tri = &gh.tri_list[tri_offset];
    sh.scounter = 0;

    auto encode_offset = sh.sm_block_id * (MAXUNDEG * NUMDIVS);
    sh.encode = &gh.encoded_induced_subgraph[encode_offset];

    auto level_offset = sh.sm_block_id * NUMDIVS * MAXDEG;
    sh.cl = &gh.level_candidate[level_offset];
    sh.pl = &gh.P[level_offset];
    sh.xl = &gh.Xp[level_offset];

    auto level_item_offset = sh.sm_block_id * MAXDEG;
    sh.level_pointer_index = &gh.level_pointer[level_item_offset];

    sh.level_pointer_index[0] = 0;
    sh.l = sh.base_l = 3;

    sh.level_pivot = 0xFFFFFFFF;

    sh.path_more_explore = false;
    sh.path_eliminated = false;
    sh.maxIntersection = 0;
    sh.Xx_sz[0] = 0;
  }
  __syncthreads();
}

template <typename T>
__device__ __forceinline__ void add_frontier(T &frontier, const T &val)
{
  if (threadIdx.x == 0)
    frontier += val;
}

template <typename T>
__device__ __forceinline__ void go_to_next_level(T &l, T &level_pointer_index)
{
  if (threadIdx.x == 0)
  {
    ++l;
    level_pointer_index = 0;
  }
}

__device__ __forceinline__ void increment_stat(uint64 &stat)
{
  if (threadIdx.x == 0)
    ++stat;
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void clean_level_L1_IP(
    GLOBAL_HANDLE<T> &gh, LOCAL_HANDLE<T> &lh, 
    SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh,
    T &level_pointer_index, T &Xx_sz)
{
  if (threadIdx.x == 0)
  {
    level_pointer_index = lh.newIndex + 1;
    sh.level_pivot = 0xFFFFFFFF;
    sh.path_more_explore = false;
    sh.maxIntersection = 0;

    sh.cnode = gh.gsplit.colInd[sh.srcStart + lh.newIndex];
    sh.cnodeStart = gh.gsplit.rowPtr[sh.cnode];
    sh.cnodeLen = gh.gsplit.splitPtr[sh.cnode] - sh.cnodeStart;
    Xx_sz = sh.Xx_aux_sz = 0;
  }
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void clean_level_L1_IPX(
    GLOBAL_HANDLE<T> &gh, LOCAL_HANDLE<T> &lh, 
    SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh,
    T &level_pointer_index, T &Xx_sz)
{
  if (threadIdx.x == 0)
  {
    level_pointer_index = lh.newIndex + 1;
    sh.level_pivot = 0xFFFFFFFF;
    sh.path_more_explore = false;
    sh.maxIntersection = 0;
    Xx_sz = sh.Xx_aux_sz = 0;
  }
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void clean_level_L2_IP(
    GLOBAL_HANDLE<T> &gh, LOCAL_HANDLE<T> &lh, 
    SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh,
    T &level_pointer_index, T &Xx_sz)
{
  if (threadIdx.x == 0)
  {
    level_pointer_index = lh.newIndex + 1;
    sh.level_pivot = 0xFFFFFFFF;
    sh.path_more_explore = false;
    sh.maxIntersection = 0;
    sh.cnode = sh.tri[lh.newIndex];
    sh.cnodeStart = gh.gsplit.rowPtr[sh.cnode];
    sh.cnodeLen = gh.gsplit.splitPtr[sh.cnode] - sh.cnodeStart;
    Xx_sz = sh.Xx_aux_sz = 0;
  }
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void clean_level_L2_IPX(
    GLOBAL_HANDLE<T> &gh, LOCAL_HANDLE<T> &lh, 
    SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh,
    T &level_pointer_index, T &Xx_sz)
{
  if (threadIdx.x == 0)
  {
    level_pointer_index = lh.newIndex + 1;
    sh.level_pivot = 0xFFFFFFFF;
    sh.path_more_explore = false;
    sh.maxIntersection = 0;
    Xx_sz = sh.Xx_aux_sz = 0;
  }
  __syncthreads();
}

template <typename T>
__device__ __forceinline__ void prepare_fork(T &Xx_aux_sz, bool &fork)
{
  if (threadIdx.x == 0)
  {
    Xx_aux_sz = 0;
    fork = false;
  }
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void try_dequeue(
    GLOBAL_HANDLE<T> &gh, SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh,
    queue_callee(queue, tickets, head, tail))
{
  if (threadIdx.x == 0 && sh.Xx_aux_sz >= 2 && sh.curSZ >= SZTH && sh.frontier >= SZFRONTIER && (*gh.work_stealing) >= gh.iteration_limit)
    queue_dequeue(queue, tickets, head, tail, CB, sh.fork, sh.worker_pos, sh.Xx_aux_sz);
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void setup_stack_donor_L1_IP(
    GLOBAL_HANDLE<T> &gh, SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh)
{
  if (threadIdx.x == 0)
  {
    size_t Xx_offset = sh.sm_block_id * MAXUNDEG;
    sh.Xx_shared = &gh.Xx[Xx_offset];
    sh.Xx_aux_shared = &gh.Xx_aux[Xx_offset];

    uint32_t *message = &gh.global_message[sh.sm_block_id * MSGCNT];
    sh.root_sm_block_id = message[0];
    sh.l = sh.base_l = message[1];
    sh.srcLen = message[2];
    sh.Xx_sz[sh.l - 2] = message[3];
    sh.i = message[4];

    sh.src = sh.i;
    sh.srcStart = gh.gsplit.splitPtr[sh.src];
    sh.srcLen = gh.gsplit.rowPtr[sh.src + 1] - sh.srcStart;
    sh.usrcStart = gh.gsplit.rowPtr[sh.src];
    sh.usrcLen = gh.gsplit.splitPtr[sh.src] - sh.usrcStart;

    size_t encode_offset = sh.root_sm_block_id * (MAXDEG * NUMDIVS);
    sh.encode = &gh.encoded_induced_subgraph[encode_offset];

    size_t level_offset = sh.sm_block_id * NUMDIVS * (MAXDEG + 1);
    sh.cl = &gh.level_candidate[level_offset];
    sh.pl = &gh.P[level_offset];
    sh.xl = &gh.Xp[level_offset];

    size_t level_item_offset = sh.sm_block_id * (MAXDEG + 1);
    sh.level_pointer_index = &gh.level_pointer[level_item_offset];

    sh.level_pointer_index[sh.l - 2] = 0;
    sh.level_pivot = 0xFFFFFFFF;

    sh.path_more_explore = false;
    sh.path_eliminated = false;
    sh.maxIntersection = 0;

    sh.num_divs_local = (sh.srcLen + 32 - 1) / 32;
    sh.lastMask_i = sh.srcLen >> 5;
    sh.lastMask_ii = (1 << (sh.srcLen & 0x1F)) - 1;
  }
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void setup_stack_donor_L1_IPX(
    GLOBAL_HANDLE<T> &gh, SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh)
{
  if (threadIdx.x == 0)
  {
    auto Xx_offset = sh.sm_block_id * MAXUNDEG;
    sh.Xx_shared = &gh.Xx[Xx_offset];
    sh.Xx_aux_shared = &gh.Xx_aux[Xx_offset];

    uint32_t *message = &gh.global_message[sh.sm_block_id * MSGCNT];
    sh.root_sm_block_id = message[0];
    sh.l = sh.base_l = message[1];
    sh.srcLen = message[2];
    sh.Xx_sz[sh.l - 2] = message[3];

    auto encode_offset = sh.root_sm_block_id * (MAXUNDEG * NUMDIVS);
    sh.encode = &gh.encoded_induced_subgraph[encode_offset];

    auto level_offset = sh.sm_block_id * NUMDIVS * (MAXDEG + 1);
    sh.cl = &gh.level_candidate[level_offset];
    sh.pl = &gh.P[level_offset];
    sh.xl = &gh.Xp[level_offset];

    auto level_item_offset = sh.sm_block_id * (MAXDEG + 1);
    sh.level_pointer_index = &gh.level_pointer[level_item_offset];

    sh.level_pointer_index[sh.l - 2] = 0;
    sh.level_pivot = 0xFFFFFFFF;

    sh.path_more_explore = false;
    sh.path_eliminated = false;
    sh.maxIntersection = 0;

    sh.num_divs_local = (sh.srcLen + 32 - 1) / 32;
    sh.lastMask_i = sh.srcLen >> 5;
    sh.lastMask_ii = (1 << (sh.srcLen & 0x1F)) - 1;
  }
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void setup_stack_donor_L2_IP(
    GLOBAL_HANDLE<T> &gh, SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh)
{
  if (threadIdx.x == 0)
  {
    auto Xx_offset = sh.sm_block_id * MAXUNDEG;
    sh.Xx_shared = &gh.Xx[Xx_offset];
    sh.Xx_aux_shared = &gh.Xx_aux[Xx_offset];

    auto tri_offset = sh.sm_block_id * MAXDEG;
    sh.tri = &gh.tri_list[tri_offset];

    uint32_t *message = &gh.global_message[sh.sm_block_id * MSGCNT];
    sh.root_sm_block_id = message[0];
    sh.l = sh.base_l = message[1];
    sh.scounter = message[2];
    sh.Xx_sz[sh.l - 3] = message[3];
    sh.i = message[4];

    sh.src = gh.gsplit.rowInd[sh.i];
    sh.srcStart = gh.gsplit.splitPtr[sh.src];
    sh.srcLen = gh.gsplit.rowPtr[sh.src + 1] - sh.srcStart;
    sh.usrcStart = gh.gsplit.rowPtr[sh.src];
    sh.usrcLen = gh.gsplit.splitPtr[sh.src] - sh.usrcStart;

    auto encode_offset = sh.root_sm_block_id * (MAXDEG * NUMDIVS);
    sh.encode = &gh.encoded_induced_subgraph[encode_offset];

    auto level_offset = sh.sm_block_id * NUMDIVS * MAXDEG;
    sh.cl = &gh.level_candidate[level_offset];
    sh.pl = &gh.P[level_offset];
    sh.xl = &gh.Xp[level_offset];

    auto level_item_offset = sh.sm_block_id * MAXDEG;
    sh.level_pointer_index = &gh.level_pointer[level_item_offset];

    sh.level_pointer_index[sh.l - 3] = 0;
    sh.level_pivot = 0xFFFFFFFF;

    sh.path_more_explore = false;
    sh.path_eliminated = false;
    sh.maxIntersection = 0;

    sh.num_divs_local = (sh.scounter + 32 - 1) / 32;
    sh.lastMask_i = sh.scounter >> 5;
    sh.lastMask_ii = (1 << (sh.scounter & 0x1F)) - 1;
  }
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void setup_stack_donor_L2_IPX(
    GLOBAL_HANDLE<T> &gh, SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh)
{
  if (threadIdx.x == 0)
  {
    auto Xx_offset = sh.sm_block_id * MAXUNDEG;
    sh.Xx_shared = &gh.Xx[Xx_offset];
    sh.Xx_aux_shared = &gh.Xx_aux[Xx_offset];

    uint32_t *message = &gh.global_message[sh.sm_block_id * MSGCNT];
    sh.root_sm_block_id = message[0];
    sh.l = sh.base_l = message[1];
    sh.scounter = message[2];
    sh.Xx_sz[sh.l - 3] = message[3];

    auto encode_offset = sh.root_sm_block_id * (MAXUNDEG * NUMDIVS);
    sh.encode = &gh.encoded_induced_subgraph[encode_offset];

    auto level_offset = sh.sm_block_id * NUMDIVS * MAXDEG;
    sh.cl = &gh.level_candidate[level_offset];
    sh.pl = &gh.P[level_offset];
    sh.xl = &gh.Xp[level_offset];

    auto level_item_offset = sh.sm_block_id * MAXDEG;
    sh.level_pointer_index = &gh.level_pointer[level_item_offset];

    sh.level_pointer_index[sh.l - 3] = 0;
    sh.level_pivot = 0xFFFFFFFF;

    sh.path_more_explore = false;
    sh.path_eliminated = false;
    sh.maxIntersection = 0;

    sh.num_divs_local = (sh.scounter + 32 - 1) / 32;
    sh.lastMask_i = sh.scounter / 32;
    sh.lastMask_ii = (1 << (sh.scounter & 0x1F)) - 1;
  }
  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void do_fork_L1(
    GLOBAL_HANDLE<T> &gh, SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh,    
    queue_callee(queue, tickets, head, tail))
{
  for (T j = 0; j < sh.Xx_aux_sz; j++)
  {
    if (threadIdx.x == 0)
      queue_wait_ticket(queue, tickets, head, tail, CB, sh.worker_pos, sh.shared_other_sm_block_id);
    __syncthreads();

    uint32_t other_sm_block_id = sh.shared_other_sm_block_id;
    auto other_level_offset = other_sm_block_id * NUMDIVS * (MAXDEG + 1);
    const T offset = sh.num_divs_local * (sh.l - 1);
    T *other_cl = &gh.level_candidate[other_level_offset];
    T *other_xl = &gh.Xp[other_level_offset];
    T *other_pl = &gh.P[other_level_offset];
    T *other_Xx = &gh.Xx[other_sm_block_id * MAXUNDEG];

    for (T k = threadIdx.x; k < sh.num_divs_local; k += blockDim.x)
    {
      other_xl[offset + k] = sh.xl[offset + k];
      other_cl[offset + k] = sh.cl[offset + k];
      other_pl[offset + k] = 0;
    }

    for (T k = threadIdx.x; k < sh.Xx_sz[sh.l - 1]; k += blockDim.x)
    {
      other_Xx[k] = sh.Xx_shared[k];
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
      const T li = offset + (sh.Xx_aux_shared[j] >> 5), ri = 1u << (sh.Xx_aux_shared[j] & 0x1F);
      uint32_t *message = &gh.global_message[other_sm_block_id * MSGCNT];
      other_pl[li] = ri;
      sh.pl[li] ^= ri;
      sh.xl[li] ^= ri;
      sh.cl[li] ^= ri;
      message[0] = sh.root_sm_block_id;
      message[1] = sh.l + 1;
      message[2] = sh.srcLen;
      message[3] = sh.Xx_sz[sh.l - 1];
      message[4] = sh.i;
      gh.work_ready[other_sm_block_id].store(1, cuda::memory_order_release);
      sh.worker_pos++;
    }
    __syncthreads();
  }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void do_fork_L2_IP(
    GLOBAL_HANDLE<T> &gh, SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh,    
    queue_callee(queue, tickets, head, tail))
{
  for (T j = 0; j < sh.Xx_aux_sz; j++)
  {
    if (threadIdx.x == 0)
      queue_wait_ticket(queue, tickets, head, tail, CB, sh.worker_pos, sh.shared_other_sm_block_id);
    __syncthreads();

    uint32_t other_sm_block_id = sh.shared_other_sm_block_id;
    auto other_level_offset = other_sm_block_id * NUMDIVS * MAXDEG;
    const T offset = sh.num_divs_local * (sh.l - 2);
    T *other_cl = &gh.level_candidate[other_level_offset];
    T *other_xl = &gh.Xp[other_level_offset];
    T *other_pl = &gh.P[other_level_offset];
    T *other_Xx = &gh.Xx[other_sm_block_id * MAXUNDEG];
    T *other_tri = &gh.tri_list[other_sm_block_id * MAXDEG];

    for (T k = threadIdx.x; k < sh.num_divs_local; k += blockDim.x)
    {
      other_xl[offset + k] = sh.xl[offset + k];
      other_cl[offset + k] = sh.cl[offset + k];
      other_pl[offset + k] = 0;
    }

    for (T k = threadIdx.x; k < sh.Xx_sz[sh.l - 2]; k += blockDim.x)
      other_Xx[k] = sh.Xx_shared[k];

    for (T k = threadIdx.x; k < sh.scounter; k += blockDim.x)
      other_tri[k] = sh.tri[k];

    __syncthreads();
    if (threadIdx.x == 0)
    {
      uint32_t *message = &gh.global_message[other_sm_block_id * MSGCNT];
      const T li = offset + (sh.Xx_aux_shared[j] >> 5), ri = 1u << (sh.Xx_aux_shared[j] & 0x1F);
      other_pl[li] = ri;
      sh.pl[li] ^= ri;
      sh.xl[li] ^= ri;
      sh.cl[li] ^= ri;
      message[0] = sh.root_sm_block_id;
      message[1] = sh.l + 1;
      message[2] = sh.scounter;
      message[3] = sh.Xx_sz[sh.l - 2];
      message[4] = sh.i;
      gh.work_ready[other_sm_block_id].store(1, cuda::memory_order_release);
      sh.worker_pos++;
    }
    __syncthreads();
  }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void do_fork_L2_IPX(
    GLOBAL_HANDLE<T> &gh, SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> &sh,    
    queue_callee(queue, tickets, head, tail))
{
  for (T j = 0; j < sh.Xx_aux_sz; j++)
  {
    if (threadIdx.x == 0)
      queue_wait_ticket(queue, tickets, head, tail, CB, sh.worker_pos, sh.shared_other_sm_block_id);
    __syncthreads();

    uint32_t other_sm_block_id = sh.shared_other_sm_block_id;
    auto other_level_offset = other_sm_block_id * NUMDIVS * MAXDEG;
    const T offset = sh.num_divs_local * (sh.l - 2);
    T *other_cl = &gh.level_candidate[other_level_offset];
    T *other_xl = &gh.Xp[other_level_offset];
    T *other_pl = &gh.P[other_level_offset];
    T *other_Xx = &gh.Xx[other_sm_block_id * MAXUNDEG];

    for (T k = threadIdx.x; k < sh.num_divs_local; k += blockDim.x)
    {
      other_xl[offset + k] = sh.xl[offset + k];
      other_cl[offset + k] = sh.cl[offset + k];
      other_pl[offset + k] = 0;
    }

    for (T k = threadIdx.x; k < sh.Xx_sz[sh.l - 2]; k += blockDim.x)
    {
      other_Xx[k] = sh.Xx_shared[k];
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
      uint32_t *message = &gh.global_message[other_sm_block_id * MSGCNT];
      const T li = offset + (sh.Xx_aux_shared[j] >> 5), ri = 1u << (sh.Xx_aux_shared[j] & 0x1F);
      other_pl[li] = ri;
      sh.pl[li] ^= ri;
      sh.xl[li] ^= ri;
      sh.cl[li] ^= ri;
      message[0] = sh.root_sm_block_id;
      message[1] = sh.l + 1;
      message[2] = sh.scounter;
      message[3] = sh.Xx_sz[sh.l - 2];
      gh.work_ready[other_sm_block_id].store(1, cuda::memory_order_release);
      sh.worker_pos++;
    }
    __syncthreads();
  }
}