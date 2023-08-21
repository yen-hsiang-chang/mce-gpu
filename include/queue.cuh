#pragma once

#include <cuda/atomic>
#include <cuda_runtime.h>

#include "cgarray.cuh"
#include "utils.cuh"

__device__ __forceinline__ uint32_t my_sleep(uint32_t ns)
{
  __nanosleep(ns);
  if (ns < (1 << 20))
    ns <<= 1;
  return ns;
}

#define queue_enqueue(queue, tickets, head, tail, queue_size, val)            \
  do                                                                          \
  {                                                                           \
    const uint32_t ticket = tail->fetch_add(1, cuda::memory_order_relaxed);   \
    const uint32_t target = ticket % queue_size;                              \
    const uint32_t ticket_target = ticket / queue_size * 2;                   \
    uint32_t ns = 8;                                                          \
    while (tickets[target].load(cuda::memory_order_relaxed) != ticket_target) \
      ns = my_sleep(ns);                                                      \
    while (tickets[target].load(cuda::memory_order_acquire) != ticket_target) \
      ns = my_sleep(ns);                                                      \
    queue[target] = val;                                                      \
    tickets[target].store(ticket_target + 1, cuda::memory_order_release);     \
  } while (0)

#define queue_dequeue(queue, tickets, head, tail, queue_size, fork, qidx, count)          \
  do                                                                                      \
  {                                                                                       \
    qidx = head->load(cuda::memory_order_relaxed);                                        \
    fork = false;                                                                         \
    if (tail->load(cuda::memory_order_relaxed) - qidx >= count)                           \
      fork = head->compare_exchange_weak(qidx, qidx + count, cuda::memory_order_relaxed); \
  } while (0)

#define queue_wait_ticket(queue, tickets, head, tail, queue_size, qidx, res)  \
  do                                                                          \
  {                                                                           \
    const uint32_t target = qidx % queue_size;                                \
    const uint32_t ticket_target = qidx / queue_size * 2 + 1;                 \
    uint32_t ns = 8;                                                          \
    while (tickets[target].load(cuda::memory_order_relaxed) != ticket_target) \
      ns = my_sleep(ns);                                                      \
    while (tickets[target].load(cuda::memory_order_acquire) != ticket_target) \
      ns = my_sleep(ns);                                                      \
    res = queue[target];                                                      \
    tickets[target].store(ticket_target + 1, cuda::memory_order_release);     \
  } while (0)

#define queue_full(queue, tickets, head, tail, queue_size) \
  (tail->load(cuda::memory_order_relaxed) - head->load(cuda::memory_order_relaxed) == queue_size)

#define queue_declare(queue, tickets, head, tail)                                                         \
  cuda::atomic<uint32_t, cuda::thread_scope_device> *tickets = nullptr, *head = nullptr, *tail = nullptr; \
  uint32_t *queue = nullptr

#define queue_init(queue, tickets, head, tail, queue_size, dev)                                                           \
  do                                                                                                                      \
  {                                                                                                                       \
    CUDA_RUNTIME(cudaMalloc((void **)&queue, queue_size * sizeof(uint32_t)));                                             \
    CUDA_RUNTIME(cudaMalloc((void **)&tickets, queue_size * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>)));  \
    CUDA_RUNTIME(cudaMemset((void *)tickets, 0, queue_size * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>))); \
    CUDA_RUNTIME(cudaMalloc((void **)&head, sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>)));                  \
    CUDA_RUNTIME(cudaMemset((void *)head, 0, sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>)));                 \
    CUDA_RUNTIME(cudaMalloc((void **)&tail, sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>)));                  \
    CUDA_RUNTIME(cudaMemset((void *)tail, 0, sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>)));                 \
  } while (0)

#define queue_free(queue, tickets, head, tail) \
  do                                           \
  {                                            \
    if (queue != nullptr)                      \
    {                                          \
      CUDA_RUNTIME(cudaFree((void *)queue));   \
      CUDA_RUNTIME(cudaFree((void *)tickets)); \
      CUDA_RUNTIME(cudaFree((void *)head));    \
      CUDA_RUNTIME(cudaFree((void *)tail));    \
    }                                          \
  } while (0)

#define queue_caller(queue, tickets, head, tail) queue, tickets, head, tail

#define queue_callee(queue, tickets, head, tail)                  \
  uint32_t *queue,                                                \
      cuda::atomic<uint32_t, cuda::thread_scope_device> *tickets, \
      cuda::atomic<uint32_t, cuda::thread_scope_device> *head,    \
      cuda::atomic<uint32_t, cuda::thread_scope_device> *tail