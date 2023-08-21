#pragma once
#include <cuda_runtime.h>

#include "cgarray.cuh"
#include "cub_wrapper.cuh"
#include "defs.h"
#include "graph_queue.cuh"
#include "logger.h"
#include "utils.cuh"

#define LEVEL_SKIP_SIZE (128)

template <typename T, typename CntType>
__global__ void init_asc(T *data, CntType count)
{
	T gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < count)
		data[gtid] = (T)gtid;
}

template <typename T, typename PeelT>
__global__ void filter_window(PeelT *edge_sup, T count, bool *in_bucket, T low, T high)
{
	auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < count)
	{
		auto v = edge_sup[gtid];
		in_bucket[gtid] = (v >= low && v < high);
	}
}

template <typename T, typename PeelT>
__global__ void filter_with_random_append(
		T *bucket_buf, T count, PeelT *EdgeSupport,
		bool *in_curr, T *curr, T *curr_cnt, T ref, T span)
{
	auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < count)
	{
		auto edge_off = bucket_buf[gtid];
		if (EdgeSupport[edge_off] >= ref && EdgeSupport[edge_off] < ref + span)
		{
			in_curr[edge_off] = true;
			auto insert_idx = atomicAdd(curr_cnt, 1);
			curr[insert_idx] = edge_off;
		}
	}
}

template <typename T, typename PeelT>
__global__ void update_priority(graph::GraphQueue_d<T, bool> current, T priority, T *nodePriority)
{
	auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < current.count[0])
	{
		auto edge_off = current.queue[gtid];
		nodePriority[edge_off] = priority;
	}
}

template <typename T>
__device__ void add_to_queue_1(graph::GraphQueue_d<T, bool> &q, T element)
{
	auto insert_idx = atomicAdd(q.count, 1);
	q.queue[insert_idx] = element;
	q.mark[element] = true;
}

template <typename T>
__device__ void add_to_queue_1_no_dup(graph::GraphQueue_d<T, bool> &q, T element)
{
	auto old_token = atomicCASBool(q.mark + element, 0, 1);
	if (!old_token)
	{
		auto insert_idx = atomicAdd(q.count, 1);
		q.queue[insert_idx] = element;
	}
}

template <typename T, typename PeelT>
__inline__ __device__ void process_degree(
		T nodeId, T level, PeelT *nodeDegree,
		graph::GraphQueue_d<T, bool> &next,
		graph::GraphQueue_d<T, bool> &bucket,
		T bucket_level_end_)
{
	auto cur = atomicSub(&nodeDegree[nodeId], 1);
	if (cur == (level + 1))
		add_to_queue_1(next, nodeId);

	// Update the Bucket.
	auto latest = cur - 1;
	if (latest > level && latest < bucket_level_end_)
		add_to_queue_1_no_dup(bucket, nodeId);
}

template <typename T, typename PeelT>
__global__ void getNodeDegree_kernel(PeelT *nodeDegree, graph::COOCSRGraph_d<T> g)
{
	uint64 gtid = threadIdx.x + blockIdx.x * blockDim.x;
	for (uint64 i = gtid; i < g.numNodes; i += blockDim.x * gridDim.x)
	{
		nodeDegree[i] = g.rowPtr[i + 1] - g.rowPtr[i];
	}
}

template <typename T, typename PeelT, int BD, int P>
__global__ void
kernel_partition_level_next(
		graph::COOCSRGraph_d<T> g,
		int level, bool *processed, PeelT *nodeDegree,
		graph::GraphQueue_d<T, bool> current,
		graph::GraphQueue_d<T, bool> &next,
		graph::GraphQueue_d<T, bool> &bucket,
		int bucket_level_end_,
		T priority,
		T *nodePriority)
{
	const size_t partitionsPerBlock = BD / P;
	const size_t lx = threadIdx.x % P;
	const int warpIdx = threadIdx.x / P; // which warp in thread block
	const size_t gwx = (blockDim.x * blockIdx.x + threadIdx.x) / P;

	for (auto i = gwx; i < current.count[0]; i += blockDim.x * gridDim.x / P)
	{
		T nodeId = current.queue[i];
		T srcStart = g.rowPtr[nodeId];
		T srcStop = g.rowPtr[nodeId + 1];

		nodePriority[nodeId] = priority;

		for (auto j = srcStart + lx; j < (srcStop + P - 1) / P * P; j += P)
		{
			__syncwarp();
			if (j < srcStop)
			{
				T affectedNode = g.colInd[j];
				if (!current.mark[affectedNode])
					process_degree<T>(affectedNode, level, nodeDegree, next, bucket, bucket_level_end_);
			}
		}
	}
}

namespace graph
{
	template <typename T, typename PeelT>
	class SingleGPU_Kcore
	{
	private:
		int dev_;
		cudaStream_t stream_;

		// Outputs:
		// Max k of a complete ktruss kernel
		int k;

		// Percentage of deleted edges for a specific k
		float percentage_deleted_k;

		// Same Function for any comutation
		void bucket_scan(
				GPUArray<PeelT> nodeDegree, T node_num, int level,
				GraphQueue<T, bool> &current,
				GPUArray<T> asc,
				GraphQueue<T, bool> &bucket,
				int &bucket_level_end_)
		{
			static bool is_first = true;
			if (is_first)
			{
				current.mark.setAll(false, true);
				bucket.mark.setAll(false, true);
				is_first = false;
			}

			const size_t block_size = 128;

			if (level == bucket_level_end_)
			{
				// Clear the bucket_removed_indicator
				T grid_size = (node_num + block_size - 1) / block_size;
				execKernel((filter_window<T, PeelT>), grid_size, block_size, dev_, false,
									 nodeDegree.gdata(), node_num, bucket.mark.gdata(), level, bucket_level_end_ + LEVEL_SKIP_SIZE);

				T val = CUBSelect(asc.gdata(), bucket.queue.gdata(), bucket.mark.gdata(), node_num, dev_);
				bucket.count.setSingle(0, val, true);
				bucket_level_end_ += LEVEL_SKIP_SIZE;
			}
			// SCAN the window.
			if (bucket.count.getSingle(0) != 0)
			{
				current.count.setSingle(0, 0, true);
				long grid_size = (bucket.count.getSingle(0) + block_size - 1) / block_size;
				execKernel((filter_with_random_append<T, PeelT>), grid_size, block_size, dev_, false,
									 bucket.queue.gdata(), bucket.count.getSingle(0), nodeDegree.gdata(), current.mark.gdata(), current.queue.gdata(), &(current.count.gdata()[0]), level, 1);
			}
			else
			{
				current.count.setSingle(0, 0, true);
			}
		}

		void AscendingGpu(int n, GPUArray<T> &identity_arr_asc)
		{
			const size_t block_size = 128;
			T grid_size = (n + block_size - 1) / block_size;
			identity_arr_asc.initialize("Identity Array Asc", gpu, n, dev_);
			execKernel(init_asc, grid_size, block_size, dev_, false, identity_arr_asc.gdata(), n);
		}

	public:
		GPUArray<PeelT> nodeDegree;
		GPUArray<T> nodePriority;

		SingleGPU_Kcore(int dev) : dev_(dev)
		{
			CUDA_RUNTIME(cudaSetDevice(dev_));
			CUDA_RUNTIME(cudaStreamCreate(&stream_));
			CUDA_RUNTIME(cudaStreamSynchronize(stream_));
		}

		SingleGPU_Kcore() : SingleGPU_Kcore(0) {}

		void getNodeDegree(COOCSRGraph_d<T> &g, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			const int dimBlock = 256;
			nodeDegree.initialize("Node Degree", gpu, g.numNodes, dev_);
			uint dimGridNodes = (g.numNodes + dimBlock - 1) / dimBlock;
			execKernel(getNodeDegree_kernel<T>, dimGridNodes, dimBlock, dev_, false, nodeDegree.gdata(), g);
		}

		void findKcoreIncremental_async(COOCSRGraph_d<T> &g, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			CUDA_RUNTIME(cudaSetDevice(dev_));
			constexpr int dimBlock = 64; // For edges and nodes

			GPUArray<BCTYPE> processed; // isDeleted

			int level = 0;
			int bucket_level_end_ = level;
			// Lets apply queues and buckets
			graph::GraphQueue<T, bool> bucket_q;
			bucket_q.Create(gpu, g.numNodes, dev_);

			graph::GraphQueue<T, bool> current_q;
			current_q.Create(gpu, g.numNodes, dev_);

			graph::GraphQueue<T, bool> next_q;
			next_q.Create(gpu, g.numNodes, dev_);
			next_q.mark.setAll(false, true);

			GPUArray<T> identity_arr_asc;
			AscendingGpu(g.numNodes, identity_arr_asc);

			nodePriority.initialize("Edge Support", gpu, g.numNodes, dev_);
			nodePriority.setAll(g.numEdges, false);

			getNodeDegree(g);
			int todo = g.numNodes;
			const auto todo_original = g.numNodes;

			T priority = 0;
			while (todo > 0)
			{
				CUDA_RUNTIME(cudaGetLastError());
				cudaDeviceSynchronize();

				// 1 bucket fill
				bucket_scan(nodeDegree, todo_original, level, current_q, identity_arr_asc, bucket_q, bucket_level_end_);

				int iterations = 0;
				while (current_q.count.getSingle(0) > 0)
				{
					todo -= current_q.count.getSingle(0);
					if (0 == todo)
						break;

					next_q.count.setSingle(0, 0, true);
					if (level == 0)
					{
						auto block_size = 256;
						auto grid_size = (current_q.count.getSingle(0) + block_size - 1) / block_size;
						execKernel((update_priority<T, PeelT>), grid_size, block_size, dev_, false, current_q.device_queue->gdata()[0], priority, nodePriority.gdata());
					}
					else
					{
						auto block_size = 256;
						auto grid_warp_size = (32 * current_q.count.getSingle(0) + block_size - 1) / block_size;
						auto grid_block_size = current_q.count.getSingle(0);
						execKernel((kernel_partition_level_next<T, PeelT, 256, 32>), grid_warp_size, block_size, dev_, false,
											 g,
											 level, processed.gdata(), nodeDegree.gdata(),
											 current_q.device_queue->gdata()[0],
											 next_q.device_queue->gdata()[0],
											 bucket_q.device_queue->gdata()[0],
											 bucket_level_end_, priority, nodePriority.gdata());
					}

					swap(current_q, next_q);
					iterations++;
					priority++;
				}
				level++;
			}

			processed.freeGPU();
			current_q.free();
			next_q.free();
			bucket_q.free();
			identity_arr_asc.freeGPU();

			k = level;

			printf("Max Core = %d\n", k - 1);
		}

		void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }
		uint count() const { return k - 1; }
		int device() const { return dev_; }
		cudaStream_t stream() const { return stream_; }
	};
} // namespace graph