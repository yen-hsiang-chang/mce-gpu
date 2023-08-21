#pragma once

#include "cgarray.cuh"
#include "defs.h"

namespace graph
{
	template <typename T, typename MarkType = bool>
	struct GraphQueue_d
	{
		T *count;
		T *queue;
		MarkType *mark;
	};

	template <typename T, typename MarkType = bool>
	class GraphQueue
	{

	public:
		GPUArray<T> count;
		GPUArray<T> queue;
		GPUArray<MarkType> mark; // mark if a node or edge is present in the graph
		GPUArray<GraphQueue_d<T, MarkType>> *device_queue;
		int capacity;

		void Create(AllocationTypeEnum at, uint cap, int devId)
		{
			capacity = cap;
			count.initialize("Queue Count", at, 1, devId);
			count.setSingle(0, 0, true);
			queue.initialize("Queue data", at, capacity, devId);
			mark.initialize("Queue Mark", at, capacity, devId);

			device_queue = new GPUArray<GraphQueue_d<T, MarkType>>();
			device_queue->initialize("Device Queue", unified, 1, devId);

			count.switch_to_gpu();
			queue.switch_to_gpu();
			mark.switch_to_gpu();

			device_queue->gdata()[0].count = count.gdata();
			device_queue->gdata()[0].queue = queue.gdata();
			device_queue->gdata()[0].mark = mark.gdata();

			device_queue->switch_to_gpu();
		}

		void CreateQueueStruct(GraphQueue_d<T, MarkType> *&d)
		{
			d = device_queue->gdata();
		}

		void free()
		{
			device_queue->freeGPU();
			count.freeGPU();
			queue.freeGPU();
			mark.freeGPU();
		}
	};
} // namespace graph