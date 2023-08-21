#include <cuda_runtime.h>
#include <iostream>
#include <omp.h>
#include <vector>

#include "../include/cgarray.cuh"
#include "../include/config.h"
#include "../include/csrcoo.cuh"
#include "../include/fileop.h"
#include "../include/kcore.cuh"
#include "../include/main_support.cuh"
#include "../include/timer.h"
#include "../include/utils.cuh"
#include "../mce/mce.cuh"

using namespace std;

int main(int argc, char **argv)
{
  Config config = parseArgs(argc, argv);

  printf("\033[0m");
  printf("Welcome ---------------------\n");
  printConfig(config);

  if (config.mt == MAINTASK::CONVERT)
  {
    graph::convert_to_bel<DataType>(config.srcGraph, config.dstGraph);
    return 0;
  }

  Timer read_graph_timer;
  vector<EdgeTy<DataType>> edges;
  graph::read_bel(config.srcGraph, edges);

  auto full = [](const EdgeTy<DataType> &e)
  { return false; };
  graph::CSRCOO<DataType> csrcoo = graph::CSRCOO<DataType>::from_edgelist(edges, full);
  vector<EdgeTy<DataType>>().swap(edges);

  DataType n = csrcoo.num_rows();
  DataType m = csrcoo.nnz();

  graph::COOCSRGraph<DataType> g;
  graph::COOCSRGraph_d<DataType> *gd = (graph::COOCSRGraph_d<DataType> *)malloc(sizeof(graph::COOCSRGraph_d<DataType>));
  g.numNodes = n;
  g.capacity = m;
  g.numEdges = m;
  gd->numNodes = g.numNodes;
  gd->numEdges = g.numEdges;
  gd->capacity = g.capacity;

  // No Allocation
  g.rowPtr = new graph::GPUArray<DataType>("Row pointer", AllocationTypeEnum::noalloc, n + 1, config.deviceId);
  g.rowInd = new graph::GPUArray<DataType>("Src Index", AllocationTypeEnum::noalloc, m, config.deviceId);
  g.colInd = new graph::GPUArray<DataType>("Dst Index", AllocationTypeEnum::noalloc, m, config.deviceId);

  DataType *rp, *ri, *ci;
  CUDA_RUNTIME(cudaMallocManaged((void **)&(rp), (n + 1) * (uint64) sizeof(DataType)));
  CUDA_RUNTIME(cudaMallocManaged((void **)&(ri), (m) * (uint64) sizeof(DataType)));
  CUDA_RUNTIME(cudaMallocManaged((void **)&(ci), (m) * (uint64) sizeof(DataType)));
  CUDA_RUNTIME(cudaMemcpy(rp, csrcoo.row_ptr(), (n + 1) * (uint64) sizeof(DataType), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaMemcpy(ri, csrcoo.row_ind(), (m) * (uint64) sizeof(DataType), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaMemcpy(ci, csrcoo.col_ind(), (m) * (uint64) sizeof(DataType), cudaMemcpyDefault));
  cudaMemAdvise(rp, (n + 1) * (uint64) sizeof(DataType), cudaMemAdviseSetReadMostly, config.deviceId /*ignored*/);
  cudaMemAdvise(ri, (m) * (uint64) sizeof(DataType), cudaMemAdviseSetReadMostly, config.deviceId /*ignored*/);
  cudaMemAdvise(ci, (m) * (uint64) sizeof(DataType), cudaMemAdviseSetReadMostly, config.deviceId /*ignored*/);
  g.rowPtr->cdata() = rp;
  g.rowPtr->setAlloc(cpuonly);
  g.rowInd->cdata() = ri;
  g.rowInd->setAlloc(cpuonly);
  g.colInd->cdata() = ci;
  g.colInd->setAlloc(cpuonly);

  Log(info, "Read graph time: %f s", read_graph_timer.elapsed());
  Log(info, "n = %u and m = %u", n, m);
  Timer transfer_timer;

  g.rowPtr->switch_to_gpu(config.deviceId);
  gd->rowPtr = g.rowPtr->gdata();

  if (g.numEdges > 1500000000)
  {
    gd->rowInd = g.rowInd->cdata();
    gd->colInd = g.colInd->cdata();
  }
  else
  {
    g.rowInd->switch_to_gpu(config.deviceId);
    g.colInd->switch_to_gpu(config.deviceId);
    gd->rowInd = g.rowInd->gdata();
    gd->colInd = g.colInd->gdata();
  }
  double transfer = transfer_timer.elapsed();
  Log(info, "Transfer Time: %f s", transfer);

  Timer degeneracy_time;
  graph::SingleGPU_Kcore<DataType, PeelType> kcore(config.deviceId);
  kcore.findKcoreIncremental_async(*gd);
  Log(info, "Degeneracy ordering time: %f s", degeneracy_time.elapsed());

  Timer csr_recreation_time;
  const size_t block_size = 1024;

  graph::COOCSRGraph_d<DataType> *gsplit = (graph::COOCSRGraph_d<DataType> *)malloc(sizeof(graph::COOCSRGraph_d<DataType>));

  gsplit->numNodes = n;
  gsplit->numEdges = m;
  gsplit->capacity = m;

  CUDA_RUNTIME(cudaMallocManaged((void **)&(gsplit->colInd), (m) * (uint64) sizeof(DataType)));
  CUDA_RUNTIME(cudaMallocManaged((void **)&(gsplit->splitPtr), (n + 1) * (uint64) sizeof(DataType)));

  CUDA_RUNTIME(cudaMallocManaged((void **)&(gsplit->rowPtr), (n + 1) * (uint64) sizeof(DataType)));
  CUDA_RUNTIME(cudaMemcpy(gsplit->rowPtr, gd->rowPtr, (n + 1) * (uint64) sizeof(DataType), cudaMemcpyDefault));

  CUDA_RUNTIME(cudaMallocManaged((void **)&(gsplit->rowInd), (m) * (uint64) sizeof(DataType)));
  CUDA_RUNTIME(cudaMemcpy(gsplit->rowInd, gd->rowInd, (m) * (uint64) sizeof(DataType), cudaMemcpyDefault));

  graph::GPUArray<DataType> tmp_block("Temp Block", AllocationTypeEnum::unified, (m + block_size - 1) / block_size, config.deviceId);
  graph::GPUArray<DataType> split_ptr("Split Ptr", AllocationTypeEnum::unified, n + 1, config.deviceId);
  tmp_block.setAll(0, true);
  split_ptr.setAll(0, true);
  execKernel(set_priority<DataType>, (m + block_size - 1) / block_size, block_size, config.deviceId, false, *gd, m, kcore.nodePriority.gdata(), tmp_block.gdata(), split_ptr.gdata());
  CUDA_RUNTIME(cudaMemcpy(gsplit->splitPtr, split_ptr.gdata(), (n + 1) * (uint64) sizeof(DataType), cudaMemcpyDefault));
  execKernel(split_pointer<DataType>, (n + 1 + block_size - 1) / block_size, block_size, config.deviceId, false, *gd, gsplit->splitPtr);
  CUBScanExclusive<DataType, DataType>(split_ptr.gdata(), split_ptr.gdata(), n + 1, config.deviceId, 0);
  CUBScanExclusive<DataType, DataType>(tmp_block.gdata(), tmp_block.gdata(), (m + block_size - 1) / block_size, config.deviceId, 0);
  execKernel((split_data<DataType, block_size>), (m + block_size - 1) / block_size, block_size, config.deviceId, false, *gd, m, kcore.nodePriority.gdata(), tmp_block.gdata(), split_ptr.gdata(), gsplit->splitPtr, gsplit->colInd);
  tmp_block.freeGPU();
  split_ptr.freeGPU();

  cudaMemAdvise(gsplit->colInd, (m) * (uint64) sizeof(DataType), cudaMemAdviseSetReadMostly, config.deviceId /*ignored*/);
  cudaMemAdvise(gsplit->splitPtr, (n + 1) * (uint64) sizeof(DataType), cudaMemAdviseSetReadMostly, config.deviceId /*ignored*/);
  cudaMemAdvise(gsplit->rowPtr, (n + 1) * (uint64) sizeof(DataType), cudaMemAdviseSetReadMostly, config.deviceId /*ignored*/);
  cudaMemAdvise(gsplit->rowInd, (m) * (uint64) sizeof(DataType), cudaMemAdviseSetReadMostly, config.deviceId /*ignored*/);

  g.rowPtr->freeGPU();
  g.rowInd->freeGPU();
  g.colInd->freeGPU();
  CUDA_RUNTIME(cudaFree(rp));
  CUDA_RUNTIME(cudaFree(ri));
  CUDA_RUNTIME(cudaFree(ci));
  free(gd);

  Log(info, "CSR Recreation time: %f s", csr_recreation_time.elapsed());

  vector<graph::MultiGPU_MCE<DataType>> mce;

  for (int i = 0; i < config.gpus.size(); i++)
    mce.push_back(graph::MultiGPU_MCE<DataType>(config.gpus[i], i, config.gpus.size()));

  Timer mce_timer;

#pragma omp parallel for
  for (int i = 0; i < config.gpus.size(); i++)
  {
    if (kcore.count() <= 300)
    {
      mce[i].mce_count<1>(*gsplit, config);
    }
    else
    {
      mce[i].mce_count<8>(*gsplit, config);
    }
    printf("Finished Launching Instance %d.\n", i);
  }
  for (int i = 0; i < config.gpus.size(); i++)
    mce[i].sync();

  double time = mce_timer.elapsed();
  Log(info, "count time %f s", time);

  uint64 tot = 0;
  for (int i = 0; i < config.gpus.size(); i++)
    tot += mce[i].show(n);
  cout << "Found " << tot << " maximal cliques in total." << '\n';

  CUDA_RUNTIME(cudaFree(gsplit->colInd));
  CUDA_RUNTIME(cudaFree(gsplit->splitPtr));
  CUDA_RUNTIME(cudaFree(gsplit->rowPtr));
  CUDA_RUNTIME(cudaFree(gsplit->rowInd));
  free(gsplit);

  printf("Done ...\n");
  return 0;
}
