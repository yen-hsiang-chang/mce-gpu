#pragma once

using uint64 = unsigned long long int;
using uint = unsigned int;
using PeelType = int;
using BCTYPE = bool;
template <typename NodeTy>
using EdgeTy = std::pair<NodeTy, NodeTy>;

using DataType = uint;

enum MAINTASK
{
  MAINTASK_UNKNOWN,
  CONVERT,
  MCE,
  MCE_LB_EVAL,
  MCE_BD_EVAL,
  MCE_DONOR_EVAL
};
enum PARLEVEL
{
  PARLEVEL_UNKNOWN,
  L1,
  L2
};
enum INDUCEDSUBGRAPH
{
  INDUCEDSUBGRAPH_UNKNOWN,
  IP,
  IPX
};
enum WORKERLIST
{
  WORKERLIST_UNKNOWN,
  NOWL,
  WL
};
enum LogPriorityEnum
{
  critical,
  warn,
  error,
  info,
  debug,
  none
};
enum AllocationTypeEnum
{
  cpuonly,
  gpu,
  unified,
  zerocopy,
  noalloc
};