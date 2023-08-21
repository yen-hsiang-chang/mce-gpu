#pragma once

#include <cstdio>
#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include "../include/cgarray.cuh"
#include "../include/config.h"
#include "../include/defs.h"
#include "../include/logger.h"
#include "../include/queue.cuh"
#include "parameter.cuh"
#include "mce_kernel_nowl.cuh"
#include "mce_kernel_wl.cuh"
#include "mce_kernel_nowl_lb.cuh"
#include "mce_kernel_wl_lb.cuh"
#include "mce_kernel_wl_bd.cuh"
#include "mce_kernel_wl_donor.cuh"
#include "mce_utils.cuh"

using namespace std;

namespace graph
{
  template <typename T, int BLOCK_DIM_X>
  __global__ void getNodeDegree_kernel(T *node_degree, graph::COOCSRGraph_d<T> g, T *max_degree)
  {
    T gtid = threadIdx.x + blockIdx.x * blockDim.x;
    typedef cub::BlockReduce<T, BLOCK_DIM_X> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T degree = 0;
    if (gtid < g.numNodes)
    {
      degree = g.rowPtr[gtid + 1] - g.rowPtr[gtid];
      node_degree[gtid] = degree;
    }

    T aggregate = BlockReduce(temp_storage).Reduce(degree, cub::Max());
    if (threadIdx.x == 0)
      atomicMax(max_degree, aggregate);
  }

  template <typename T, int BLOCK_DIM_X>
  __global__ void getSplitDegree_kernel(T *node_degree, graph::COOCSRGraph_d<T> g, T *max_degree)
  {
    T gtid = threadIdx.x + blockIdx.x * blockDim.x;
    typedef cub::BlockReduce<T, BLOCK_DIM_X> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T degree = 0;
    if (gtid < g.numNodes)
    {
      degree = g.rowPtr[gtid + 1] - g.splitPtr[gtid];
      node_degree[gtid] = degree;
    }

    T aggregate = BlockReduce(temp_storage).Reduce(degree, cub::Max());
    if (threadIdx.x == 0)
      atomicMax(max_degree, aggregate);
  }

  template <typename T>
  class MultiGPU_MCE
  {
  private:
    int dev_, global_id_, total_instance_;
    cudaStream_t stream_;

  public:
    GPUArray<T> node_degree, max_degree, max_undirected_degree;
    GPUArray<uint64> mce_counter;
    GPUArray<T> Xx, Xx_aux, level_candidate, encoded_induced_subgraph;
    GPUArray<T> P, Xp, level_pointer, tri_list, work_stealing;
    GPUArray<uint32_t> global_message;
    cuda::atomic<uint32_t, cuda::thread_scope_device> *work_ready = nullptr;
    queue_declare(queue, tickets, head, tail);

    MultiGPU_MCE(int dev, int global_id, int total_instance) 
      : dev_(dev), global_id_(global_id), total_instance_(total_instance)
    {
      CUDA_RUNTIME(cudaSetDevice(dev_));
      CUDA_RUNTIME(cudaStreamCreate(&stream_));
      CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    }

    MultiGPU_MCE() : MultiGPU_MCE(0, 0, 0) {}

    void getNodeDegree(COOCSRGraph_d<T> &g, T *maxD)
    {
      const int dimBlock = 128;
      node_degree.initialize("Edge Support", unified, g.numNodes, dev_);
      uint dimGridNodes = (g.numNodes + dimBlock - 1) / dimBlock;
      execKernel((getNodeDegree_kernel<T, dimBlock>),
                 dimGridNodes, dimBlock, dev_, false, 
                 node_degree.gdata(), g, maxD);
    }

    void getSplitDegree(COOCSRGraph_d<T> &g, T *maxD)
    {
      const int dimBlock = 128;
      node_degree.initialize("Edge Support", unified, g.numNodes, dev_);
      uint dimGridNodes = (g.numNodes + dimBlock - 1) / dimBlock;
      execKernel((getSplitDegree_kernel<T, dimBlock>),
                 dimGridNodes, dimBlock, dev_, false,
                 node_degree.gdata(), g, maxD);
    }

    template <const int PSIZE>
    void mce_count(COOCSRGraph_d<T> &gsplit, Config config)
    {
      CUDA_RUNTIME(cudaSetDevice(dev_));

      mce_counter = GPUArray<uint64>("maximal clique count", unified, 1, dev_);
      mce_counter.setAll(0, true);

      max_degree.initialize("degree", unified, 1, dev_);
      max_degree.setSingle(0, 0, true);
      getSplitDegree(gsplit, max_degree.gdata());

      max_undirected_degree.initialize("undirected degree", unified, 1, dev_);
      max_undirected_degree.setSingle(0, 0, true);
      getNodeDegree(gsplit, max_undirected_degree.gdata());

      CUDAContext context;
      T num_SMs = context.num_SMs;
      const uint block_size = 
        (uint64)max_degree.gdata()[0] * max_undirected_degree.gdata()[0] >= 1e8 ? 256 : 128;
      T conc_blocks_per_SM = context.GetConCBlocks(block_size);
      const uint partition_size = PSIZE;
      const uint dv = 32;
      const uint max_level = max_degree.gdata()[0] + 1;
      const uint num_divs = (max_degree.gdata()[0] + dv - 1) / dv;
      const uint64 encode_size = 
        config.induced == INDUCEDSUBGRAPH::IP ? 
        num_SMs * conc_blocks_per_SM * max_degree.gdata()[0] * num_divs :
        num_SMs * conc_blocks_per_SM * max_undirected_degree.gdata()[0] * num_divs;
      encoded_induced_subgraph.initialize("induced subgraph", gpu, encode_size, dev_);
      const uint64 level_size = num_SMs * conc_blocks_per_SM * max_level * num_divs;
      const uint64 level_item_size = num_SMs * conc_blocks_per_SM * max_level;

      level_candidate.initialize("level candidate", gpu, level_size, dev_);
      P.initialize("P(possible)", gpu, level_size, dev_);
      Xp.initialize("X from P", gpu, level_size, dev_);

      level_pointer.initialize("level pointer", gpu, level_item_size, dev_);
      level_candidate.setAll(0, true);
      encoded_induced_subgraph.setAll(0, true);

      const uint64 tri_size = num_SMs * conc_blocks_per_SM * max_degree.gdata()[0];
      tri_list.initialize("triangle list", gpu, tri_size, dev_);
      tri_list.setAll(0, true);

      const uint numPartitions = block_size / partition_size;
      const uint msg_cnt = 5;
      const uint conc_blocks = num_SMs * conc_blocks_per_SM;
      cudaMemcpyToSymbol(PARTSIZE, &partition_size, sizeof(PARTSIZE));
      cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART));
      cudaMemcpyToSymbol(MAXLEVEL, &max_level, sizeof(MAXLEVEL));
      cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
      cudaMemcpyToSymbol(MAXDEG, &(max_degree.gdata()[0]), sizeof(MAXDEG));
      cudaMemcpyToSymbol(MAXUNDEG, &(max_undirected_degree.gdata()[0]), sizeof(MAXDEG));
      cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM), sizeof(CBPSM));
      cudaMemcpyToSymbol(MSGCNT, &(msg_cnt), sizeof(MSGCNT));
      cudaMemcpyToSymbol(CB, &(conc_blocks), sizeof(CB));

      CUDA_RUNTIME(cudaMalloc((void **)&work_ready, conc_blocks * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>)));
      CUDA_RUNTIME(cudaMemset((void *)work_ready, 0, conc_blocks * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>)));
      global_message = GPUArray<uint32_t>("global message", gpu, conc_blocks * msg_cnt, dev_);
      queue_init(queue, tickets, head, tail, conc_blocks, dev_);
      work_stealing = GPUArray<uint32_t>("work stealing counter", gpu, 1, dev_);
      work_stealing.setAll(global_id_, true);

      Xx = GPUArray<T>("X for X", gpu, max_undirected_degree.gdata()[0] * num_SMs * conc_blocks_per_SM, dev_);
      Xx_aux = GPUArray<T>("auxiliary array for X for X ", gpu, 
                           max_undirected_degree.gdata()[0] * num_SMs * conc_blocks_per_SM, dev_);
      Xx.setAll(0, true);

      cudaDeviceSynchronize();
      CUDA_RUNTIME(cudaGetLastError());
      auto grid_block_size = num_SMs * conc_blocks_per_SM;
      auto kernel = mce_kernel_l2_ip_nowl<T, 128, partition_size>;
      const uint kid = (block_size == 128 ? 0 : 1) + (config.workerlist == WORKERLIST::WL ? 0 : 2) +
                       (config.induced == INDUCEDSUBGRAPH::IP ? 0 : 4) + (config.level == PARLEVEL::L1 ? 0 : 8) +
                       (config.mt == MAINTASK::MCE ? 0 : (config.mt == MAINTASK::MCE_LB_EVAL ? 16 : (config.mt == MAINTASK::MCE_BD_EVAL ? 32 : 48)));
      switch (kid)
      {
      case 0:
        kernel = mce_kernel_l1_ip_wl<T, 128, partition_size>;
        break;
      case 1:
        kernel = mce_kernel_l1_ip_wl<T, 256, partition_size>;
        break;
      case 2:
        kernel = mce_kernel_l1_ip_nowl<T, 128, partition_size>;
        break;
      case 3:
        kernel = mce_kernel_l1_ip_nowl<T, 256, partition_size>;
        break;
      case 4:
        kernel = mce_kernel_l1_ipx_wl<T, 128, partition_size>;
        break;
      case 5:
        kernel = mce_kernel_l1_ipx_wl<T, 256, partition_size>;
        break;
      case 6:
        kernel = mce_kernel_l1_ipx_nowl<T, 128, partition_size>;
        break;
      case 7:
        kernel = mce_kernel_l1_ipx_nowl<T, 256, partition_size>;
        break;
      case 8:
        kernel = mce_kernel_l2_ip_wl<T, 128, partition_size>;
        break;
      case 9:
        kernel = mce_kernel_l2_ip_wl<T, 256, partition_size>;
        break;
      case 10:
        kernel = mce_kernel_l2_ip_nowl<T, 128, partition_size>;
        break;
      case 11:
        kernel = mce_kernel_l2_ip_nowl<T, 256, partition_size>;
        break;
      case 12:
        kernel = mce_kernel_l2_ipx_wl<T, 128, partition_size>;
        break;
      case 13:
        kernel = mce_kernel_l2_ipx_wl<T, 256, partition_size>;
        break;
      case 14:
        kernel = mce_kernel_l2_ipx_nowl<T, 128, partition_size>;
        break;
      case 15:
        kernel = mce_kernel_l2_ipx_nowl<T, 256, partition_size>;
        break;

      case 16:
        kernel = mce_kernel_l1_ip_wl_lb<T, 128, partition_size>;
        break;
      case 17:
        kernel = mce_kernel_l1_ip_wl_lb<T, 256, partition_size>;
        break;
      case 18:
        kernel = mce_kernel_l1_ip_nowl_lb<T, 128, partition_size>;
        break;
      case 19:
        kernel = mce_kernel_l1_ip_nowl_lb<T, 256, partition_size>;
        break;
      case 20:
        kernel = mce_kernel_l1_ipx_wl_lb<T, 128, partition_size>;
        break;
      case 21:
        kernel = mce_kernel_l1_ipx_wl_lb<T, 256, partition_size>;
        break;
      case 22:
        kernel = mce_kernel_l1_ipx_nowl_lb<T, 128, partition_size>;
        break;
      case 23:
        kernel = mce_kernel_l1_ipx_nowl_lb<T, 256, partition_size>;
        break;
      case 24:
        kernel = mce_kernel_l2_ip_wl_lb<T, 128, partition_size>;
        break;
      case 25:
        kernel = mce_kernel_l2_ip_wl_lb<T, 256, partition_size>;
        break;
      case 26:
        kernel = mce_kernel_l2_ip_nowl_lb<T, 128, partition_size>;
        break;
      case 27:
        kernel = mce_kernel_l2_ip_nowl_lb<T, 256, partition_size>;
        break;
      case 28:
        kernel = mce_kernel_l2_ipx_wl_lb<T, 128, partition_size>;
        break;
      case 29:
        kernel = mce_kernel_l2_ipx_wl_lb<T, 256, partition_size>;
        break;
      case 30:
        kernel = mce_kernel_l2_ipx_nowl_lb<T, 128, partition_size>;
        break;
      case 31:
        kernel = mce_kernel_l2_ipx_nowl_lb<T, 256, partition_size>;
        break;

      case 32:
        kernel = mce_kernel_l1_ip_wl_bd<T, 128, partition_size>;
        break;
      case 33:
        kernel = mce_kernel_l1_ip_wl_bd<T, 256, partition_size>;
        break;
      case 34:
        assert(0);
        break;
      case 35:
        assert(0);
        break;
      case 36:
        kernel = mce_kernel_l1_ipx_wl_bd<T, 128, partition_size>;
        break;
      case 37:
        kernel = mce_kernel_l1_ipx_wl_bd<T, 256, partition_size>;
        break;
      case 38:
        assert(0);
        break;
      case 39:
        assert(0);
        break;
      case 40:
        kernel = mce_kernel_l2_ip_wl_bd<T, 128, partition_size>;
        break;
      case 41:
        kernel = mce_kernel_l2_ip_wl_bd<T, 256, partition_size>;
        break;
      case 42:
        assert(0);
        break;
      case 43:
        assert(0);
        break;
      case 44:
        kernel = mce_kernel_l2_ipx_wl_bd<T, 128, partition_size>;
        break;
      case 45:
        kernel = mce_kernel_l2_ipx_wl_bd<T, 256, partition_size>;
        break;
      case 46:
        assert(0);
        break;
      case 47:
        assert(0);
        break;
      
      case 48:
        kernel = mce_kernel_l1_ip_wl_donor<T, 128, partition_size>;
        break;
      case 49:
        kernel = mce_kernel_l1_ip_wl_donor<T, 256, partition_size>;
        break;
      case 50:
        assert(0);
        break;
      case 51:
        assert(0);
        break;
      case 52:
        kernel = mce_kernel_l1_ipx_wl_donor<T, 128, partition_size>;
        break;
      case 53:
        kernel = mce_kernel_l1_ipx_wl_donor<T, 256, partition_size>;
        break;
      case 54:
        assert(0);
        break;
      case 55:
        assert(0);
        break;
      case 56:
        kernel = mce_kernel_l2_ip_wl_donor<T, 128, partition_size>;
        break;
      case 57:
        kernel = mce_kernel_l2_ip_wl_donor<T, 256, partition_size>;
        break;
      case 58:
        assert(0);
        break;
      case 59:
        assert(0);
        break;
      case 60:
        kernel = mce_kernel_l2_ipx_wl_donor<T, 128, partition_size>;
        break;
      case 61:
        kernel = mce_kernel_l2_ipx_wl_donor<T, 256, partition_size>;
        break;
      case 62:
        assert(0);
        break;
      case 63:
        assert(0);
        break;
      default:
        assert(0);
      };

      if (config.mt == MAINTASK::MCE_LB_EVAL)
        printf("BLOCK ID, SM ID, Number of traversed nodes\n");
      if (config.mt == MAINTASK::MCE_BD_EVAL)
        printf("OTHER, BUILDING INDUCED SUBGRAPHS, WORKER LIST, SELECTING PIVOTS, SET OPERATIONS, TESTING MAXIMALITY, BRANCHING AND BACKTRACKING\n");
      if (config.mt == MAINTASK::MCE_DONOR_EVAL)
        printf("BLOCK ID, SM ID, Number of donations\n");

      GLOBAL_HANDLE<T> gh;
      gh.gsplit = gsplit;
      gh.iteration_limit = config.level == PARLEVEL::L1 ? gsplit.numNodes : gsplit.numEdges;
      gh.level_candidate = level_candidate.gdata();
      gh.mce_counter = mce_counter.gdata();
      gh.encoded_induced_subgraph = encoded_induced_subgraph.gdata();
      gh.P = P.gdata();

      gh.Xp = Xp.gdata();
      gh.level_pointer = level_pointer.gdata();
      gh.Xx = Xx.gdata();
      gh.Xx_aux = Xx_aux.gdata();
      gh.tri_list = tri_list.gdata();
      gh.work_ready = work_ready;
      gh.global_message = global_message.gdata();
      gh.work_stealing = work_stealing.gdata();
      gh.stride = total_instance_;
      execKernelAsync((kernel),
                      grid_block_size, block_size, dev_, stream_, false,
                      gh, queue_caller(queue, tickets, head, tail));

      if (config.mt != MAINTASK::MCE)
        sync();
    }

    uint64 show(const T &n)
    {
      cout.imbue(std::locale(""));
      cout << "Instance " << global_id_ << " found " << mce_counter.gdata()[0] << " maximal cliques.\n";
      return mce_counter.gdata()[0];
    }

    ~MultiGPU_MCE()
    {
      free_memory();
    }

    void free_memory()
    {
      level_candidate.freeGPU();
      encoded_induced_subgraph.freeGPU();
      P.freeGPU();
      Xp.freeGPU();
      level_pointer.freeGPU();
      Xx_aux.freeGPU();
      tri_list.freeGPU();
      node_degree.freeGPU();
      Xx.freeGPU();
      max_degree.freeGPU();
      max_undirected_degree.freeGPU();
      mce_counter.freeGPU();

      if (work_ready != nullptr)
      {
        CUDA_RUNTIME(cudaFree((void *)work_ready));
      }
      global_message.freeGPU();
      queue_free(queue, tickets, head, tail);
    }

    void sync()
    {
      CUDA_RUNTIME(cudaSetDevice(dev_));
      cudaDeviceSynchronize();
    }
    int device() const { return dev_; }
    cudaStream_t stream() const { return stream_; }
  };
}