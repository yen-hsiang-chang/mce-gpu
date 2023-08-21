#pragma once
#include "../include/defs.h"
#include "../include/queue.cuh"
#include "../include/utils.cuh"
#include "mce_utils.cuh"
#include "parameter.cuh"

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 2048 / BLOCK_DIM_X)
__global__ void mce_kernel_l1_ip_nowl_lb(
  GLOBAL_HANDLE<T> gh,
  queue_callee(queue, tickets, head, tail))
{
  LOCAL_HANDLE<T> lh;
  __shared__ SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> sh;
  __shared__ uint64 traversed_nodes;
  
  lh.numPartitions = BLOCK_DIM_X / CPARTSIZE;
  lh.wx = threadIdx.x / CPARTSIZE;
  lh.lx = threadIdx.x % CPARTSIZE;
  lh.partMask = (CPARTSIZE == 32 ? 0xFFFFFFFF : (1 << CPARTSIZE) - 1) 
                  << ((lh.wx % (32 / CPARTSIZE)) * CPARTSIZE);

  if (threadIdx.x == 0)
  {
    sh.root_sm_block_id = sh.sm_block_id = blockIdx.x;
    sh.state = 0;
    sh.count = 0;
    traversed_nodes = 0;
  }
  __syncthreads();

  while (sh.state != 100)
  {
    __syncthreads();
    if (sh.state == 0)
    {
      if (threadIdx.x == 0)
        sh.i = atomicAdd((T *)gh.work_stealing, gh.stride);
      __syncthreads();

      if (sh.i >= gh.iteration_limit)
        break;

      setup_stack_original_L1_IP(gh, sh);
      encode_clear(lh, sh, sh.srcLen);

      for (T j = lh.wx; j < sh.srcLen; j += lh.numPartitions)
      {
        auto &g = gh.gsplit;
        graph::warp_sorted_count_and_encode_full<T, CPARTSIZE>(
          &g.colInd[sh.srcStart], sh.srcLen,
          &g.colInd[g.splitPtr[g.colInd[sh.srcStart + j]]], 
          g.rowPtr[g.colInd[sh.srcStart + j] + 1] 
            - g.splitPtr[g.colInd[sh.srcStart + j]],
          j, sh.num_divs_local, sh.encode
        );
      }
      __syncthreads();

      init_max_count_and_index(lh, sh, sh.srcLen);
      select_pivot_from_P(lh, sh, sh.srcLen);

      if (sh.path_more_explore)
      {
        finalize_pivot(lh, sh, sh.srcLen);
        apply_pivot_to_first_level(sh);
      }
      else
      {
        continue;
      }
    }

    __syncthreads();

    while (sh.l >= sh.base_l)
    {
      if (!get_next_candidate(gh, lh, sh, sh.srcLen, 
                              sh.level_pointer_index[sh.l - 2],
                              sh.pl + sh.num_divs_local * (sh.l - 2)))
        continue;

      increment_stat(traversed_nodes);
      clean_level_L1_IP(gh, lh, sh, sh.level_pointer_index[sh.l - 2], 
                        sh.Xx_sz[sh.l - 1]);
      Xx_compaction_for_IP(gh, sh, sh.l - 2);
      populate_xl_and_cl(lh, sh, sh.l - 2);
      init_max_count_and_index(lh, sh, sh.srcLen);
      get_candidate_size(lh, sh);
      select_pivot_from_P_and_Xp(lh, sh, sh.srcLen);

      if (!sh.path_more_explore || sh.path_eliminated)
      {
        test_maximality(sh, sh.Xx_sz[sh.l - 1]);
      }
      else
      {
        finalize_pivot(lh, sh, sh.srcLen);
        apply_pivot_to_next_level<false>(sh, sh.l - 1);
        go_to_next_level(sh.l, sh.level_pointer_index[sh.l - 1]);
      }
      __syncthreads();
    }
    __syncthreads();
  }
  __syncthreads();

  if (threadIdx.x == 0)
  {
    atomicAdd(gh.mce_counter, sh.count);
    printf("%u, %u, %llu\n", blockIdx.x, __mysmid(), traversed_nodes);
  }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 2048 / BLOCK_DIM_X)
__global__ void mce_kernel_l1_ipx_nowl_lb(
  GLOBAL_HANDLE<T> gh,
  queue_callee(queue, tickets, head, tail))
{
   LOCAL_HANDLE<T> lh;
  __shared__ SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> sh;
  __shared__ uint64 traversed_nodes;

  lh.numPartitions = BLOCK_DIM_X / CPARTSIZE;
  lh.wx = threadIdx.x / CPARTSIZE;
  lh.lx = threadIdx.x % CPARTSIZE;
  lh.partMask = (CPARTSIZE == 32 ? 0xFFFFFFFF : (1 << CPARTSIZE) - 1) 
                  << ((lh.wx % (32 / CPARTSIZE)) * CPARTSIZE);

  if (threadIdx.x == 0)
  {
    sh.root_sm_block_id = sh.sm_block_id = blockIdx.x;
    sh.state = 0;
    sh.count = 0;
    traversed_nodes = 0;
  }
  __syncthreads();

  while (sh.state != 100)
  {
    __syncthreads();
    if (sh.state == 0)
    {
      if (threadIdx.x == 0)
        sh.i = atomicAdd((T *)gh.work_stealing, gh.stride);
      __syncthreads();

      if (sh.i >= gh.iteration_limit)
        break; 

      setup_stack_original_L1_IPX(gh, sh);
      encode_clear(lh, sh, sh.usrcLen + sh.srcLen);

      for (T j = lh.wx; j < sh.usrcLen + sh.srcLen; j += lh.numPartitions)
      {
        auto &g = gh.gsplit;
        graph::warp_sorted_count_and_encode_full_mclique<T, CPARTSIZE>(
          &g.colInd[sh.srcStart], sh.srcLen,
          &g.colInd[g.splitPtr[g.colInd[sh.usrcStart + j]]],
          g.rowPtr[g.colInd[sh.usrcStart + j] + 1] 
            - g.splitPtr[g.colInd[sh.usrcStart + j]],
          j, sh.num_divs_local, sh.encode, sh.usrcLen);
      }
      __syncthreads();

      init_max_count_and_index(lh, sh, sh.srcLen);
      select_pivot_from_P(lh, sh, sh.srcLen);
      select_pivot_from_Xx_first_level(lh, sh, sh.Xx_sz[0], sh.srcLen);

      if (sh.path_more_explore && !sh.path_eliminated)
      {
        finalize_pivot(lh, sh, sh.srcLen);
        apply_pivot_to_first_level(sh);
      }
      else
      {
        continue;
      }
    }

    __syncthreads();
    while (sh.l >= sh.base_l)
    {
      if (!get_next_candidate(gh, lh, sh, sh.srcLen, 
                              sh.level_pointer_index[sh.l - 2],
                              sh.pl + sh.num_divs_local * (sh.l - 2)))
        continue;

      increment_stat(traversed_nodes);
      clean_level_L1_IPX(gh, lh, sh, sh.level_pointer_index[sh.l - 2], sh.Xx_sz[sh.l - 1]);
      Xx_compaction_for_IPX(lh, sh, sh.l - 2);
      populate_xl_and_cl(lh, sh, sh.l - 2);
      init_max_count_and_index(lh, sh, sh.srcLen);
      get_candidate_size(lh, sh);
      select_pivot_from_Xx_not_first_level(lh, sh, sh.Xx_sz[sh.l - 1], sh.srcLen);
      select_pivot_from_P_and_Xp(lh, sh, sh.srcLen);

      if (!sh.path_more_explore || sh.path_eliminated)
      {
        test_maximality(sh, sh.Xx_sz[sh.l - 1]);
      }
      else
      {
        finalize_pivot(lh, sh, sh.srcLen);
        apply_pivot_to_next_level<false>(sh, sh.l - 1);
        go_to_next_level(sh.l, sh.level_pointer_index[sh.l - 1]);
      }
      __syncthreads();
    }
    __syncthreads();
  }
  __syncthreads();

  if (threadIdx.x == 0)
  {
    atomicAdd(gh.mce_counter, sh.count);
    printf("%u, %u, %llu\n", blockIdx.x, __mysmid(), traversed_nodes);
  }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 2048 / BLOCK_DIM_X)
__global__ void mce_kernel_l2_ip_nowl_lb(
  GLOBAL_HANDLE<T> gh,
  queue_callee(queue, tickets, head, tail))
{
  LOCAL_HANDLE<T> lh;
  __shared__ SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> sh;
  __shared__ uint64 traversed_nodes;

  lh.numPartitions = BLOCK_DIM_X / CPARTSIZE;
  lh.wx = threadIdx.x / CPARTSIZE;
  lh.lx = threadIdx.x % CPARTSIZE;
  lh.partMask = (CPARTSIZE == 32 ? 0xFFFFFFFF : (1 << CPARTSIZE) - 1) 
                  << ((lh.wx % (32 / CPARTSIZE)) * CPARTSIZE);

  if (threadIdx.x == 0)
  {
    sh.root_sm_block_id = sh.sm_block_id = blockIdx.x;
    sh.state = 0;
    sh.count = 0;
    traversed_nodes = 0;
  }
  __syncthreads();

  while (sh.state != 100)
  {
    __syncthreads();
    if (sh.state == 0)
    {
      if (threadIdx.x == 0)
        sh.i = atomicAdd((T *)gh.work_stealing, gh.stride);
      __syncthreads();

      if (sh.i >= gh.iteration_limit)
      {
        break;
      }

      if (sh.i < gh.gsplit.splitPtr[gh.gsplit.rowInd[sh.i]])
        continue;

      setup_stack_original_L2_IP(gh, sh);

      for (T j = threadIdx.x; j < sh.usrcLen + sh.srcLen; j += BLOCK_DIM_X)
      {
        bool found = false;
        T cur = gh.gsplit.colInd[sh.usrcStart + j];
        graph::binary_search(gh.gsplit.colInd + sh.cnodeStart, 0u, sh.cnodeLen, cur, found);
        if (found)
          sh.Xx_shared[atomicAdd(&sh.Xx_sz[0], 1)] = j;
      }
      __syncthreads();

      graph::block_count_and_set_tri<BLOCK_DIM_X, T>(
        &gh.gsplit.colInd[sh.srcStart], sh.srcLen,
        &gh.gsplit.colInd[sh.src2Start], sh.src2Len,
        sh.tri, &sh.scounter);

      __syncthreads();

      if (threadIdx.x == 0)
      {
        sh.num_divs_local = (sh.scounter + 32 - 1) / 32;
        sh.lastMask_i = sh.scounter / 32;
        sh.lastMask_ii = (1 << (sh.scounter & 0x1F)) - 1;
      }
      __syncthreads();

      encode_clear(lh, sh, sh.scounter);

      for (T j = lh.wx; j < sh.scounter; j += lh.numPartitions)
      {
        auto &g = gh.gsplit;
        graph::warp_sorted_count_and_encode_full<T, CPARTSIZE>(
          sh.tri, sh.scounter,
          &g.colInd[g.splitPtr[sh.tri[j]]],
          g.rowPtr[sh.tri[j] + 1] - g.splitPtr[sh.tri[j]],
          j, sh.num_divs_local, sh.encode);
      }
      __syncthreads();

      init_max_count_and_index(lh, sh, sh.scounter);
      select_pivot_from_P(lh, sh, sh.scounter);

      if (sh.path_more_explore)
      {
        finalize_pivot(lh, sh, sh.scounter);
        apply_pivot_to_first_level(sh);
      }
      else
      {
        if (threadIdx.x == 0 && sh.Xx_sz[0] == 0)
          ++sh.count;
        continue;
      }
    }

    __syncthreads();

    while (sh.l >= sh.base_l)
    {
      if (!get_next_candidate(gh, lh, sh, sh.scounter, 
                              sh.level_pointer_index[sh.l - 3],
                              sh.pl + sh.num_divs_local * (sh.l - 3)))
        continue;

      increment_stat(traversed_nodes);
      clean_level_L2_IP(gh, lh, sh, sh.level_pointer_index[sh.l - 3], 
                           sh.Xx_sz[sh.l - 2]);
      Xx_compaction_for_IP(gh, sh, sh.l - 3);
      populate_xl_and_cl(lh, sh, sh.l - 3);
      init_max_count_and_index(lh, sh, sh.scounter);
      get_candidate_size(lh ,sh);
      select_pivot_from_P_and_Xp(lh, sh, sh.scounter);

      if (!sh.path_more_explore || sh.path_eliminated)
      {
        test_maximality(sh, sh.Xx_sz[sh.l - 2]);
      }
      else
      {
        finalize_pivot(lh, sh, sh.scounter);
        apply_pivot_to_next_level<false>(sh, sh.l - 2);
        go_to_next_level(sh.l, sh.level_pointer_index[sh.l - 2]);
      }
      __syncthreads();
    }
    __syncthreads();
  }
  __syncthreads();

  if (threadIdx.x == 0)
  {
    atomicAdd(gh.mce_counter, sh.count);
    printf("%u, %u, %llu\n", blockIdx.x, __mysmid(), traversed_nodes);
  }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 2048 / BLOCK_DIM_X)
__global__ void mce_kernel_l2_ipx_nowl_lb(
  GLOBAL_HANDLE<T> gh,
  queue_callee(queue, tickets, head, tail))
{
  LOCAL_HANDLE<T> lh;
  __shared__ SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> sh;
  __shared__ uint64 traversed_nodes;

  lh.numPartitions = BLOCK_DIM_X / CPARTSIZE;
  lh.wx = threadIdx.x / CPARTSIZE;
  lh.lx = threadIdx.x % CPARTSIZE;
  lh.partMask = (CPARTSIZE == 32 ? 0xFFFFFFFF : (1 << CPARTSIZE) - 1) 
                  << ((lh.wx % (32 / CPARTSIZE)) * CPARTSIZE);

  if (threadIdx.x == 0)
  {
    sh.root_sm_block_id = sh.sm_block_id = blockIdx.x;
    sh.state = 0;
    sh.count = 0;
    traversed_nodes = 0;
  }
  __syncthreads();

  while (sh.state != 100)
  {
    __syncthreads();
    if (sh.state == 0)
    {
      if (threadIdx.x == 0)
        sh.i = atomicAdd((T *)gh.work_stealing, gh.stride);
      __syncthreads();

      if (sh.i >= gh.iteration_limit)
      {
        break;
      }

      if (sh.i < gh.gsplit.splitPtr[gh.gsplit.rowInd[sh.i]])
        continue;

      setup_stack_original_L2_IPX(gh, sh);

      for (T j = threadIdx.x; j < sh.usrcLen + sh.srcLen; j += BLOCK_DIM_X)
      {
        bool found = false;
        T cur = gh.gsplit.colInd[sh.usrcStart + j];
        graph::binary_search(gh.gsplit.colInd + sh.cnodeStart, 0u, sh.cnodeLen, cur, found);
        if (found)
          sh.Xx_shared[atomicAdd(&sh.Xx_sz[0], 1)] = sh.usrcStart + j;
      }
      __syncthreads();

      graph::block_count_and_set_tri<BLOCK_DIM_X, T>(
        &gh.gsplit.colInd[sh.srcStart], sh.srcLen,
        &gh.gsplit.colInd[sh.src2Start], sh.src2Len,
        sh.tri, &sh.scounter);

      __syncthreads();

      if (threadIdx.x == 0)
      {
        sh.num_divs_local = (sh.scounter + 32 - 1) / 32;
        sh.lastMask_i = sh.scounter / 32;
        sh.lastMask_ii = (1 << (sh.scounter & 0x1F)) - 1;
      }
      __syncthreads();

      encode_clear(lh, sh, sh.scounter + sh.Xx_sz[0]);

      for (T j = lh.wx; j < sh.scounter; j += lh.numPartitions)
      {
        auto &g = gh.gsplit;
        graph::warp_sorted_count_and_encode_full_mclique<T, CPARTSIZE>(
          sh.tri, sh.scounter,
          &g.colInd[g.splitPtr[sh.tri[j]]], 
          g.rowPtr[sh.tri[j] + 1] - g.splitPtr[sh.tri[j]],
          j + sh.Xx_sz[0], sh.num_divs_local, sh.encode, sh.Xx_sz[0]);
      }

      for (T j = lh.wx; j < sh.Xx_sz[0]; j += lh.numPartitions)
      {
        auto &g = gh.gsplit;
        graph::warp_sorted_count_and_encode_full_mclique<T, CPARTSIZE>(
          sh.tri, sh.scounter,
          &g.colInd[g.splitPtr[g.colInd[sh.Xx_shared[j]]]],
          g.rowPtr[g.colInd[sh.Xx_shared[j]] + 1] 
            - g.splitPtr[g.colInd[sh.Xx_shared[j]]],
          j, sh.num_divs_local, sh.encode, sh.Xx_sz[0]);
      }
      __syncthreads();

      for (T j = threadIdx.x; j < sh.Xx_sz[0]; j += BLOCK_DIM_X)
      {
        sh.Xx_shared[j] = j + sh.scounter;
      }
      __syncthreads();

      init_max_count_and_index(lh, sh, sh.scounter);
      select_pivot_from_P(lh, sh, sh.scounter);
      select_pivot_from_Xx_first_level(lh, sh, sh.Xx_sz[0], sh.scounter);

      if (sh.path_more_explore && !sh.path_eliminated)
      {
        finalize_pivot(lh, sh, sh.scounter);
        apply_pivot_to_first_level(sh);
      }
      else
      {
        if (threadIdx.x == 0 && sh.Xx_sz[0] == 0)
          ++sh.count;
        continue;
      }
    }

    __syncthreads();

    while (sh.l >= sh.base_l)
    {
      if (!get_next_candidate(gh, lh, sh, sh.scounter, 
                              sh.level_pointer_index[sh.l - 3],
                              sh.pl + sh.num_divs_local * (sh.l - 3)))
        continue;

      increment_stat(traversed_nodes);
      clean_level_L2_IPX(gh, lh, sh, sh.level_pointer_index[sh.l - 3], sh.Xx_sz[sh.l - 2]);
      Xx_compaction_for_IPX(lh, sh, sh.l - 3);
      populate_xl_and_cl(lh, sh, sh.l - 3);
      init_max_count_and_index(lh, sh, sh.scounter);
      get_candidate_size(lh, sh);
      select_pivot_from_Xx_not_first_level(lh, sh, sh.Xx_sz[sh.l - 2], sh.scounter);
      select_pivot_from_P_and_Xp(lh, sh, sh.scounter);

      if (!sh.path_more_explore || sh.path_eliminated)
      {
        test_maximality(sh, sh.Xx_sz[sh.l - 2]);
      }
      else
      {
        finalize_pivot(lh, sh, sh.scounter);
        apply_pivot_to_next_level<false>(sh, sh.l - 2);
        go_to_next_level(sh.l, sh.level_pointer_index[sh.l - 2]);
      }
      __syncthreads();
    }
    __syncthreads();
  }
  __syncthreads();

  if (threadIdx.x == 0)
  {
    atomicAdd(gh.mce_counter, sh.count);
    printf("%u, %u, %llu\n", blockIdx.x, __mysmid(), traversed_nodes);
  }
}