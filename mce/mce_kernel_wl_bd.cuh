#pragma once
#include "../include/defs.h"
#include "../include/queue.cuh"
#include "../include/utils.cuh"
#include "mce_utils.cuh"
#include "parameter.cuh"

enum CounterName
{
  OTHER = 0,
  INDUCED,
  QUEUEOP,
  PIVOT,
  INTERSECTION,
  MAXIMALITY,
  TREE,
  NUM_COUNTERS
};

struct Counters
{
  unsigned long long tmp[NUM_COUNTERS];
  unsigned long long totalTime[NUM_COUNTERS];
};

static __device__ void initializeCounters(Counters *counters)
{
  __syncthreads();
  if (threadIdx.x == 0)
  {
    for (unsigned int i = 0; i < NUM_COUNTERS; ++i)
    {
      counters->totalTime[i] = 0;
    }
  }
  __syncthreads();
}

static __device__ void startTime(CounterName counterName, Counters *counters, bool sync = false)
{
  if(sync)  __syncthreads();
  if (threadIdx.x == 0)
  {
    counters->tmp[counterName] = clock64();
  }
  if(sync)  __syncthreads();
}

static __device__ void endTime(CounterName counterName, Counters *counters, bool sync = true)
{
  if(sync)  __syncthreads();
  if (threadIdx.x == 0)
  {
    counters->totalTime[counterName] += clock64() - counters->tmp[counterName];
  }
  if(sync)  __syncthreads();
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 2048 / BLOCK_DIM_X)
__global__ void mce_kernel_l1_ip_wl_bd(
  GLOBAL_HANDLE<T> gh,
  queue_callee(queue, tickets, head, tail))
{
  LOCAL_HANDLE<T> lh;
  __shared__ SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> sh;
  __shared__ Counters timer;

  initializeCounters(&timer);

  startTime(OTHER, &timer);  
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
    sh.frontier = 0;
  }
  __syncthreads();
  endTime(OTHER, &timer);

  while (sh.state != 100)
  {
    __syncthreads();
    if (sh.state == 0)
    {
      startTime(OTHER, &timer);
      if (threadIdx.x == 0)
        sh.i = atomicAdd((T *)gh.work_stealing, gh.stride);
      __syncthreads();
      endTime(OTHER, &timer);

      if (sh.i >= gh.iteration_limit)
      {
        startTime(QUEUEOP, &timer);
        __syncthreads();
        if (threadIdx.x == 0)
        {
          sh.state = 1;
          queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
        }
        endTime(QUEUEOP, &timer);
        continue;
      }

      startTime(OTHER, &timer);
      setup_stack_original_L1_IP(gh, sh);
      endTime(OTHER, &timer);

      startTime(INDUCED, &timer);
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
      endTime(INDUCED, &timer);

      startTime(PIVOT, &timer);
      init_max_count_and_index(lh, sh, sh.srcLen);
      select_pivot_from_P(lh, sh, sh.srcLen);
      endTime(PIVOT, &timer);

      if (sh.path_more_explore)
      {
        startTime(PIVOT, &timer);
        finalize_pivot(lh, sh, sh.srcLen);
        add_frontier<T>(sh.frontier, sh.srcLen - sh.maxIntersection);
        apply_pivot_to_first_level(sh);
        endTime(PIVOT, &timer);
      }
      else
      {
        continue;
      }
    }
    else if (sh.state == 1)
    {
      __syncthreads();
      if (threadIdx.x == 0)
      {
        uint32_t ns = 8;
        do
        {
          startTime(QUEUEOP, &timer, false);
          if (gh.work_ready[sh.sm_block_id].load(cuda::memory_order_relaxed))
          {
            if (gh.work_ready[sh.sm_block_id].load(cuda::memory_order_acquire))
            {
              sh.state = 2;
              gh.work_ready[sh.sm_block_id].store(0, cuda::memory_order_relaxed);
              endTime(QUEUEOP, &timer, false);
              break;
            }
          }
          else if (queue_full(queue, tickets, head, tail, CB))
          {
            sh.state = 100;
            endTime(QUEUEOP, &timer, false);
            break;
          }
          endTime(QUEUEOP, &timer, false);
        } while (ns = my_sleep(ns));
      }
      __syncthreads();
      continue;
    }
    else if (sh.state == 2)
    {
      __syncthreads();
      startTime(OTHER, &timer);
      add_frontier<T>(sh.frontier, 1);
      setup_stack_donor_L1_IP(gh, sh);
      endTime(OTHER, &timer);
    }
    __syncthreads();

    while (sh.l >= sh.base_l)
    {
      startTime(TREE, &timer);
      if (!get_next_candidate(gh, lh, sh, sh.srcLen, 
                              sh.level_pointer_index[sh.l - 2],
                              sh.pl + sh.num_divs_local * (sh.l - 2)))
      {
        endTime(TREE, &timer);
        continue;
      }
      endTime(TREE, &timer);

      startTime(INTERSECTION, &timer);
      add_frontier<T>(sh.frontier, -1);
      clean_level_L1_IP(gh, lh, sh, sh.level_pointer_index[sh.l - 2], 
                        sh.Xx_sz[sh.l - 1]);
      Xx_compaction_for_IP(gh, sh, sh.l - 2);
      populate_xl_and_cl(lh, sh, sh.l - 2);
      endTime(INTERSECTION, &timer);

      startTime(PIVOT, &timer);
      init_max_count_and_index(lh, sh, sh.srcLen);
      get_candidate_size(lh, sh);
      select_pivot_from_P_and_Xp(lh, sh, sh.srcLen);
      endTime(PIVOT, &timer);

      if (!sh.path_more_explore || sh.path_eliminated)
      {
        startTime(MAXIMALITY, &timer);
        test_maximality(sh, sh.Xx_sz[sh.l - 1]);
        endTime(MAXIMALITY, &timer);
      }
      else
      {
        startTime(PIVOT, &timer);
        finalize_pivot(lh, sh, sh.srcLen);
        prepare_fork(sh.Xx_aux_sz, sh.fork);
        apply_pivot_to_next_level<true>(sh, sh.l - 1);
        endTime(PIVOT, &timer);
        
        startTime(QUEUEOP, &timer);
        try_dequeue(gh, sh, queue_caller(queue, tickets, head, tail));
        endTime(QUEUEOP, &timer);

        if (sh.fork)
        {
          startTime(QUEUEOP, &timer);
          do_fork_L1(gh, sh, queue_caller(queue, tickets, head, tail));
          endTime(QUEUEOP, &timer);
        }
        else
        {
          startTime(TREE, &timer);
          add_frontier<T>(sh.frontier, sh.Xx_aux_sz);
          go_to_next_level(sh.l, sh.level_pointer_index[sh.l - 1]);
          endTime(TREE, &timer);
        }
        __syncthreads();
      }
      __syncthreads();
    }

    __syncthreads();
    startTime(QUEUEOP, &timer);
    if (threadIdx.x == 0 && sh.state == 2)
    {
      sh.state = 1;
      queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
    }
    endTime(QUEUEOP, &timer);
    __syncthreads();
  }
  __syncthreads();

  if (threadIdx.x == 0)
  {
    atomicAdd(gh.mce_counter, sh.count);
    assert(NUM_COUNTERS == 7);
    printf("%llu, %llu, %llu, %llu, %llu, %llu, %llu\n",
           timer.totalTime[0], timer.totalTime[1], timer.totalTime[2], timer.totalTime[3],
           timer.totalTime[4], timer.totalTime[5], timer.totalTime[6]);
    assert(sh.frontier == 0);
  }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 2048 / BLOCK_DIM_X)
__global__ void mce_kernel_l1_ipx_wl_bd(
  GLOBAL_HANDLE<T> gh,
  queue_callee(queue, tickets, head, tail))
{
  LOCAL_HANDLE<T> lh;
  __shared__ SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> sh;
  __shared__ Counters timer;

  initializeCounters(&timer);

  startTime(OTHER, &timer);

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
    sh.frontier = 0;
  }
  __syncthreads();
  endTime(OTHER, &timer);

  while (sh.state != 100)
  {
    __syncthreads();
    if (sh.state == 0)
    {
      startTime(OTHER, &timer);
      if (threadIdx.x == 0)
        sh.i = atomicAdd((T *)gh.work_stealing, gh.stride);
      __syncthreads();
      endTime(OTHER, &timer);

      if (sh.i >= gh.iteration_limit)
      {
        __syncthreads();
        startTime(QUEUEOP, &timer);
        if (threadIdx.x == 0)
        {
          sh.state = 1;
          queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
        }
        endTime(QUEUEOP, &timer);
        continue;
      }

      startTime(OTHER, &timer);
      setup_stack_original_L1_IPX(gh, sh);
      endTime(OTHER, &timer);

      startTime(INDUCED, &timer);
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
      endTime(INDUCED, &timer);

      startTime(PIVOT, &timer);
      init_max_count_and_index(lh, sh, sh.srcLen);
      select_pivot_from_P(lh, sh, sh.srcLen);
      select_pivot_from_Xx_first_level(lh, sh, sh.Xx_sz[0], sh.srcLen);
      endTime(PIVOT, &timer);

      if (sh.path_more_explore && !sh.path_eliminated)
      {
        startTime(PIVOT, &timer);
        finalize_pivot(lh, sh, sh.srcLen);
        add_frontier<T>(sh.frontier, sh.srcLen - sh.maxIntersection);
        apply_pivot_to_first_level(sh);
        endTime(PIVOT, &timer);
      }
      else
      {
        continue;
      }
    }
    else if (sh.state == 1)
    {
      __syncthreads();
      if (threadIdx.x == 0)
      {
        uint32_t ns = 8;
        do
        {
          startTime(QUEUEOP, &timer, false);
          if (gh.work_ready[sh.sm_block_id].load(cuda::memory_order_relaxed))
          {
            if (gh.work_ready[sh.sm_block_id].load(cuda::memory_order_acquire))
            {
              sh.state = 2;
              gh.work_ready[sh.sm_block_id].store(0, cuda::memory_order_relaxed);
              endTime(QUEUEOP, &timer, false);
              break;
            }
          }
          else if (queue_full(queue, tickets, head, tail, CB))
          {
            sh.state = 100;
            endTime(QUEUEOP, &timer, false);
            break;
          }
          endTime(QUEUEOP, &timer, false);
        } while (ns = my_sleep(ns));
      }
      __syncthreads();
      continue;
    }
    else if (sh.state == 2)
    {
      __syncthreads();
      startTime(OTHER, &timer);
      add_frontier<T>(sh.frontier, 1);
      setup_stack_donor_L1_IPX(gh, sh);
      endTime(OTHER, &timer);
    }

    __syncthreads();
    while (sh.l >= sh.base_l)
    {
      startTime(TREE, &timer);
      if (!get_next_candidate(gh, lh, sh, sh.srcLen, 
                              sh.level_pointer_index[sh.l - 2],
                              sh.pl + sh.num_divs_local * (sh.l - 2)))
      {
        endTime(TREE, &timer);
        continue;
      }
      endTime(TREE, &timer);

      startTime(INTERSECTION, &timer);
      add_frontier<T>(sh.frontier, -1);
      clean_level_L1_IPX(gh, lh, sh, sh.level_pointer_index[sh.l - 2], sh.Xx_sz[sh.l - 1]);
      Xx_compaction_for_IPX(lh, sh, sh.l - 2);
      populate_xl_and_cl(lh, sh, sh.l - 2);
      endTime(INTERSECTION, &timer);

      startTime(PIVOT, &timer);
      init_max_count_and_index(lh, sh, sh.srcLen);
      get_candidate_size(lh, sh);
      select_pivot_from_Xx_not_first_level(lh, sh, sh.Xx_sz[sh.l - 1], sh.srcLen);
      select_pivot_from_P_and_Xp(lh, sh, sh.srcLen);
      endTime(PIVOT, &timer);

      if (!sh.path_more_explore || sh.path_eliminated)
      {
        startTime(MAXIMALITY, &timer);
        test_maximality(sh, sh.Xx_sz[sh.l - 1]);
        endTime(MAXIMALITY, &timer);
      }
      else
      {
        startTime(PIVOT, &timer);
        finalize_pivot(lh, sh, sh.srcLen);
        prepare_fork(sh.Xx_aux_sz, sh.fork);
        apply_pivot_to_next_level<true>(sh, sh.l - 1);
        endTime(PIVOT, &timer);

        startTime(QUEUEOP, &timer);
        try_dequeue(gh, sh, queue_caller(queue, tickets, head, tail));
        endTime(QUEUEOP, &timer);

        if (sh.fork)
        {
          startTime(QUEUEOP, &timer);
          do_fork_L1(gh, sh, queue_caller(queue, tickets, head, tail));
          endTime(QUEUEOP, &timer);
        }
        else
        {
          startTime(TREE, &timer);
          add_frontier<T>(sh.frontier, sh.Xx_aux_sz);
          go_to_next_level(sh.l, sh.level_pointer_index[sh.l - 1]);
          endTime(TREE, &timer);
        }
        __syncthreads();
      }
      __syncthreads();
    }

    __syncthreads();
    startTime(QUEUEOP, &timer);
    if (threadIdx.x == 0 && sh.state == 2)
    {
      sh.state = 1;
      queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
    }
    endTime(QUEUEOP, &timer);
    __syncthreads();
  }
  __syncthreads();

  if (threadIdx.x == 0)
  {
    atomicAdd(gh.mce_counter, sh.count);
    assert(NUM_COUNTERS == 7);
    printf("%llu, %llu, %llu, %llu, %llu, %llu, %llu\n",
           timer.totalTime[0], timer.totalTime[1], timer.totalTime[2], timer.totalTime[3],
           timer.totalTime[4], timer.totalTime[5], timer.totalTime[6]);
    assert(sh.frontier == 0);
  }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 2048 / BLOCK_DIM_X)
__global__ void mce_kernel_l2_ip_wl_bd(
  GLOBAL_HANDLE<T> gh,
  queue_callee(queue, tickets, head, tail))
{
  LOCAL_HANDLE<T> lh;
  __shared__ SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> sh;
  __shared__ Counters timer;

  initializeCounters(&timer);

  startTime(OTHER, &timer);

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
    sh.frontier = 0;
  }
  __syncthreads();
  endTime(OTHER, &timer);

  while (sh.state != 100)
  {
    __syncthreads();
    if (sh.state == 0)
    {
      startTime(OTHER, &timer);
      if (threadIdx.x == 0)
        sh.i = atomicAdd((T *)gh.work_stealing, gh.stride);
      __syncthreads();
      endTime(OTHER, &timer);

      if (sh.i >= gh.iteration_limit)
      {
        __syncthreads();
        startTime(QUEUEOP, &timer);
        if (threadIdx.x == 0)
        {
          sh.state = 1;
          queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
        }
        endTime(QUEUEOP, &timer);
        continue;
      }

      if (sh.i < gh.gsplit.splitPtr[gh.gsplit.rowInd[sh.i]])
        continue;

      startTime(OTHER, &timer);
      setup_stack_original_L2_IP(gh, sh);
      endTime(OTHER, &timer);

      startTime(INDUCED, &timer);
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
      endTime(INDUCED, &timer);

      startTime(PIVOT, &timer);
      init_max_count_and_index(lh, sh, sh.scounter);
      select_pivot_from_P(lh, sh, sh.scounter);
      endTime(PIVOT, &timer);

      if (sh.path_more_explore)
      {
        startTime(PIVOT, &timer);
        finalize_pivot(lh, sh, sh.scounter);
        add_frontier<T>(sh.frontier, sh.scounter - sh.maxIntersection);
        apply_pivot_to_first_level(sh);
        endTime(PIVOT, &timer);
      }
      else
      {
        startTime(MAXIMALITY, &timer);
        if (threadIdx.x == 0 && sh.Xx_sz[0] == 0)
          ++sh.count;
        endTime(MAXIMALITY, &timer);
        continue;
      }
    }
    else if (sh.state == 1)
    {
      __syncthreads();
      if (threadIdx.x == 0)
      {
        uint32_t ns = 8;
        do
        {
          startTime(QUEUEOP, &timer, false);
          if (gh.work_ready[sh.sm_block_id].load(cuda::memory_order_relaxed))
          {
            if (gh.work_ready[sh.sm_block_id].load(cuda::memory_order_acquire))
            {
              sh.state = 2;
              gh.work_ready[sh.sm_block_id].store(0, cuda::memory_order_relaxed);
              endTime(QUEUEOP, &timer, false);
              break;
            }
          }
          else if (queue_full(queue, tickets, head, tail, CB))
          {
            sh.state = 100;
            endTime(QUEUEOP, &timer, false);
            break;
          }
          endTime(QUEUEOP, &timer, false);
        } while (ns = my_sleep(ns));
      }
      __syncthreads();
      continue;
    }
    else if (sh.state == 2)
    {
      startTime(OTHER, &timer);
      __syncthreads();
      add_frontier<T>(sh.frontier, 1);
      setup_stack_donor_L2_IP<T>(gh, sh);
      endTime(OTHER, &timer);
    }

    __syncthreads();

    while (sh.l >= sh.base_l)
    {
      startTime(TREE, &timer);
      if (!get_next_candidate(gh, lh, sh, sh.scounter, 
                              sh.level_pointer_index[sh.l - 3],
                              sh.pl + sh.num_divs_local * (sh.l - 3)))
      {
        endTime(TREE, &timer);
        continue;
      }
      endTime(TREE, &timer);

      startTime(INTERSECTION, &timer);
      add_frontier<T>(sh.frontier, -1);
      clean_level_L2_IP(gh, lh, sh, sh.level_pointer_index[sh.l - 3], 
                           sh.Xx_sz[sh.l - 2]);
      Xx_compaction_for_IP(gh, sh, sh.l - 3);
      populate_xl_and_cl(lh, sh, sh.l - 3);
      endTime(INTERSECTION, &timer);

      startTime(PIVOT, &timer);
      init_max_count_and_index(lh, sh, sh.scounter);
      get_candidate_size(lh ,sh);
      select_pivot_from_P_and_Xp(lh, sh, sh.scounter);
      endTime(PIVOT, &timer);

      if (!sh.path_more_explore || sh.path_eliminated)
      {
        startTime(MAXIMALITY, &timer);
        test_maximality(sh, sh.Xx_sz[sh.l - 2]);
        endTime(MAXIMALITY, &timer);
      }
      else
      {
        startTime(PIVOT, &timer);
        finalize_pivot(lh, sh, sh.scounter);
        prepare_fork(sh.Xx_aux_sz, sh.fork);
        apply_pivot_to_next_level<true>(sh, sh.l - 2);
        endTime(PIVOT, &timer);

        startTime(QUEUEOP, &timer);
        try_dequeue(gh, sh, queue_caller(queue, tickets, head, tail));
        endTime(QUEUEOP, &timer);

        if (sh.fork)
        {
          startTime(QUEUEOP, &timer);
          do_fork_L2_IP(gh, sh, queue_caller(queue, tickets, head, tail));
          endTime(QUEUEOP, &timer);
        }
        else
        {
          startTime(TREE, &timer);
          add_frontier<T>(sh.frontier, sh.Xx_aux_sz);
          go_to_next_level(sh.l, sh.level_pointer_index[sh.l - 2]);
          endTime(TREE, &timer);
        }
        __syncthreads();
      }
      __syncthreads();
    }

    __syncthreads();
    startTime(QUEUEOP, &timer);
    if (threadIdx.x == 0 && sh.state == 2)
    {
      sh.state = 1;
      queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
    }
    endTime(QUEUEOP, &timer);
    __syncthreads();
  }
  __syncthreads();

  if (threadIdx.x == 0)
  {
    atomicAdd(gh.mce_counter, sh.count);
    assert(NUM_COUNTERS == 7);
    printf("%llu, %llu, %llu, %llu, %llu, %llu, %llu\n",
           timer.totalTime[0], timer.totalTime[1], timer.totalTime[2], timer.totalTime[3],
           timer.totalTime[4], timer.totalTime[5], timer.totalTime[6]);
    assert(sh.frontier == 0);
  }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 2048 / BLOCK_DIM_X)
__global__ void mce_kernel_l2_ipx_wl_bd(
  GLOBAL_HANDLE<T> gh,
  queue_callee(queue, tickets, head, tail))
{
  LOCAL_HANDLE<T> lh;
  __shared__ SHARED_HANDLE<T, BLOCK_DIM_X, CPARTSIZE> sh;
  __shared__ Counters timer;

  initializeCounters(&timer);

  startTime(OTHER, &timer);

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
    sh.frontier = 0;
  }
  __syncthreads();
  endTime(OTHER, &timer);

  while (sh.state != 100)
  {
    __syncthreads();
    if (sh.state == 0)
    {
      startTime(OTHER, &timer);
      if (threadIdx.x == 0)
        sh.i = atomicAdd((T *)gh.work_stealing, gh.stride);
      __syncthreads();
      endTime(OTHER, &timer);

      if (sh.i >= gh.iteration_limit)
      {
        __syncthreads();
        startTime(QUEUEOP, &timer);
        if (threadIdx.x == 0)
        {
          sh.state = 1;
          queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
        }
        endTime(QUEUEOP, &timer);
        continue;
      }

      if (sh.i < gh.gsplit.splitPtr[gh.gsplit.rowInd[sh.i]])
        continue;

      startTime(OTHER, &timer);
      setup_stack_original_L2_IPX(gh, sh);
      endTime(OTHER, &timer);

      startTime(INDUCED, &timer);
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
      endTime(INDUCED, &timer);

      startTime(PIVOT, &timer);
      init_max_count_and_index(lh, sh, sh.scounter);
      select_pivot_from_P(lh, sh, sh.scounter);
      select_pivot_from_Xx_first_level(lh, sh, sh.Xx_sz[0], sh.scounter);
      endTime(PIVOT, &timer);

      if (sh.path_more_explore && !sh.path_eliminated)
      {
        startTime(PIVOT, &timer);
        finalize_pivot(lh, sh, sh.scounter);
        add_frontier<T>(sh.frontier, sh.scounter - sh.maxIntersection);
        apply_pivot_to_first_level(sh);
        endTime(PIVOT, &timer);
      }
      else
      {
        startTime(MAXIMALITY, &timer);
        if (threadIdx.x == 0 && sh.Xx_sz[0] == 0)
          ++sh.count;
        endTime(MAXIMALITY, &timer);
        continue;
      }
    }
    else if (sh.state == 1)
    {
      __syncthreads();
      if (threadIdx.x == 0)
      {
        uint32_t ns = 8;
        do
        {
          startTime(QUEUEOP, &timer, false);
          if (gh.work_ready[sh.sm_block_id].load(cuda::memory_order_relaxed))
          {
            if (gh.work_ready[sh.sm_block_id].load(cuda::memory_order_acquire))
            {
              sh.state = 2;
              gh.work_ready[sh.sm_block_id].store(0, cuda::memory_order_relaxed);
              endTime(QUEUEOP, &timer, false);
              break;
            }
          }
          else if (queue_full(queue, tickets, head, tail, CB))
          {
            sh.state = 100;
            endTime(QUEUEOP, &timer, false);
            break;
          }
          endTime(QUEUEOP, &timer, false);
        } while (ns = my_sleep(ns));
      }
      __syncthreads();
      continue;
    }
    else if (sh.state == 2)
    {
      __syncthreads();
      startTime(OTHER, &timer);
      add_frontier<T>(sh.frontier, 1);
      setup_stack_donor_L2_IPX(gh, sh);
      endTime(OTHER, &timer);
    }

    __syncthreads();

    while (sh.l >= sh.base_l)
    {
      startTime(TREE, &timer);
      if (!get_next_candidate(gh, lh, sh, sh.scounter, 
                              sh.level_pointer_index[sh.l - 3],
                              sh.pl + sh.num_divs_local * (sh.l - 3)))
      {
        endTime(TREE, &timer);
        continue;
      }
      endTime(TREE, &timer);

      startTime(INTERSECTION, &timer);
      add_frontier<T>(sh.frontier, -1);
      clean_level_L2_IPX(gh, lh, sh, sh.level_pointer_index[sh.l - 3], sh.Xx_sz[sh.l - 2]);
      Xx_compaction_for_IPX(lh, sh, sh.l - 3);
      populate_xl_and_cl(lh, sh, sh.l - 3);
      endTime(INTERSECTION, &timer);

      startTime(PIVOT, &timer);
      init_max_count_and_index(lh, sh, sh.scounter);
      get_candidate_size(lh, sh);
      select_pivot_from_Xx_not_first_level(lh, sh, sh.Xx_sz[sh.l - 2], sh.scounter);
      select_pivot_from_P_and_Xp(lh, sh, sh.scounter);
      endTime(PIVOT, &timer);

      if (!sh.path_more_explore || sh.path_eliminated)
      {
        startTime(MAXIMALITY, &timer);
        test_maximality(sh, sh.Xx_sz[sh.l - 2]);
        endTime(MAXIMALITY, &timer);
      }
      else
      {
        startTime(PIVOT, &timer);
        finalize_pivot(lh, sh, sh.scounter);
        prepare_fork(sh.Xx_aux_sz, sh.fork);
        apply_pivot_to_next_level<true>(sh, sh.l - 2);
        endTime(PIVOT, &timer);

        startTime(QUEUEOP, &timer);
        try_dequeue(gh, sh, queue_caller(queue, tickets, head, tail));
        endTime(QUEUEOP, &timer);

        if (sh.fork)
        {
          startTime(QUEUEOP, &timer);
          do_fork_L2_IPX(gh, sh, queue_caller(queue, tickets, head, tail));
          endTime(QUEUEOP, &timer);
        }
        else
        {
          startTime(TREE, &timer);
          add_frontier<T>(sh.frontier, sh.Xx_aux_sz);
          go_to_next_level(sh.l, sh.level_pointer_index[sh.l - 2]);
          endTime(TREE, &timer);
        }
        __syncthreads();
      }
      __syncthreads();
    }

    startTime(QUEUEOP, &timer);
    __syncthreads();
    if (threadIdx.x == 0 && sh.state == 2)
    {
      sh.state = 1;
      queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
    }
    endTime(QUEUEOP, &timer);
    __syncthreads();
  }
  __syncthreads();

  if (threadIdx.x == 0)
  {
    atomicAdd(gh.mce_counter, sh.count);
    assert(NUM_COUNTERS == 7);
    printf("%llu, %llu, %llu, %llu, %llu, %llu, %llu\n",
           timer.totalTime[0], timer.totalTime[1], timer.totalTime[2], timer.totalTime[3],
           timer.totalTime[4], timer.totalTime[5], timer.totalTime[6]);
    assert(sh.frontier == 0);
  }
}