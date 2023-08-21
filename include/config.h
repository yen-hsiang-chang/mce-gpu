#pragma once
#include "defs.h"
#include <unistd.h>
#include <vector>
#include <cctype>
using namespace std;

struct Config
{
  string srcGraph;
  string dstGraph;
  MAINTASK mt;
  PARLEVEL level;
  INDUCEDSUBGRAPH induced;
  WORKERLIST workerlist;
  int deviceId;
  vector<int> gpus;
};

static MAINTASK parseMainTask(const string &s)
{
  if (s == "convert")
    return MAINTASK::CONVERT;
  if (s == "mce")
    return MAINTASK::MCE;
  if (s == "mce-lb-eval")
    return MAINTASK::MCE_LB_EVAL;
  if (s == "mce-bd-eval")
    return MAINTASK::MCE_BD_EVAL;
  if (s == "mce-donor-eval")
    return MAINTASK::MCE_DONOR_EVAL;
  fprintf(stderr, "Unrecognized -m option (Main Task): %s\n", s.c_str());
  exit(0);
  return MAINTASK::MAINTASK_UNKNOWN;
}

static string asString(const MAINTASK &mt)
{
  if (mt == MAINTASK::CONVERT)
    return "convert";
  if (mt == MAINTASK::MCE)
    return "mce";
  if (mt == MAINTASK::MCE_LB_EVAL)
    return "mce-lb-eval";
  if (mt == MAINTASK::MCE_BD_EVAL)
    return "mce-bd-eval";
  if (mt == MAINTASK::MCE_DONOR_EVAL)
    return "mce-donor-eval";
  return "unknown";
}

static PARLEVEL parseLevel(const string &s)
{
  if (s == "l1")
    return PARLEVEL::L1;
  if (s == "l2")
    return PARLEVEL::L2;
  fprintf(stderr, "Unrecognized -p option (Parallelization Scheme): %s\n", s.c_str());
  exit(0);
  return PARLEVEL::PARLEVEL_UNKNOWN;
}

static string asString(const PARLEVEL &l)
{
  if (l == PARLEVEL::L1)
    return "L1";
  if (l == PARLEVEL::L2)
    return "L2";
  return "unknown";
}

static INDUCEDSUBGRAPH parseInduced(const string &s)
{
  if (s == "p")
    return INDUCEDSUBGRAPH::IP;
  if (s == "px")
    return INDUCEDSUBGRAPH::IPX;
  fprintf(stderr, "Unrecognized -i option (Induced Subgraphs Scheme): %s\n", s.c_str());
  exit(0);
  return INDUCEDSUBGRAPH::INDUCEDSUBGRAPH_UNKNOWN;
}

static string asString(const INDUCEDSUBGRAPH &induced)
{
  if (induced == INDUCEDSUBGRAPH::IP)
    return "IP";
  if (induced == INDUCEDSUBGRAPH::IPX)
    return "IPX";
  return "unknown";
}

static WORKERLIST parseWorkerList(const string &s)
{
  if (s == "nowl")
    return WORKERLIST::NOWL;
  if (s == "wl")
    return WORKERLIST::WL;
  fprintf(stderr, "Unrecognized -w option (Worker List Scheme): %s\n", s.c_str());
  exit(0);
  return WORKERLIST::WORKERLIST_UNKNOWN;
}

static string asString(const WORKERLIST &workerlist)
{
  if (workerlist == WORKERLIST::NOWL)
    return "No Worker List";
  if (workerlist == WORKERLIST::WL)
    return "Worker List";
  return "unknown";
}

static std::vector<int> parseDevice(const string &s)
{
  std::vector<int> res;
  const int lim = s.size();
  for (int i = 0; i < lim; i++)
  {
    if (!isdigit(s[i]))
      continue;
    int cur = s[i] - '0', j = i;
    while (j + 1 < lim && isdigit(s[j + 1]))
      cur = cur * 10 + s[++j] - '0';
    res.emplace_back(cur);
    i = j + 1;
  }
  return res;
}

static string asString(const vector<int> &v)
{
  string res = "";
  for (int i = 0; i < v.size(); i++)
  {
    if (i != 0)
      res += ",";
    res += to_string(v[i]);
  }
  return res;
}

static void usage()
{
  fprintf(stderr,
          "\nUsage:  ./parallel_mce_on_gpus [options]"
          "\n"
          "\nOptions:"
          "\n    -g <Src graph FileName>       Name of file with source graph"
          "\n    -r <Dst graph FileName>       Name of file with destination graph only for conversion"
          "\n    -d <Device Id(s)>             GPU Device Id(s) separated by commas without spaces"
          "\n    -m <Main Task>                Name of the task to perform <convert: graph conversion, mce, mce-lb-eval: load balance evaluation, mce-bd-eval: breakdown evaluation, mce-donor-eval: donation evaluation>"
          "\n    -p <Parallelization Scheme>   Level of subtrees to parallelize <l1: first level, l2: second level>"
          "\n    -i <Induced Subgraphs Scheme> Building induced subgraphs from which sets <p: P only, px: P and X>"
          "\n    -w <Worker List Scheme>       Use worker list to achieve load balance or not <nowl: No worker list, wl: Use worker list>"
          "\n    -h                            Help"
          "\n"
          "\n");
}

static Config parseArgs(int argc, char **argv)
{
  Config config;
  config.srcGraph = "";
  config.dstGraph = "";
  config.mt = MAINTASK::MAINTASK_UNKNOWN;
  config.level = PARLEVEL::PARLEVEL_UNKNOWN;
  config.induced = INDUCEDSUBGRAPH::INDUCEDSUBGRAPH_UNKNOWN;
  config.workerlist = WORKERLIST::WORKERLIST_UNKNOWN;
  config.deviceId = 0;
  config.gpus = std::vector<int>();

  int opt;

  while ((opt = getopt(argc, argv, "g:r:d:m:p:i:w:h")) >= 0)
  {
    switch (opt)
    {
    case 'g':
      config.srcGraph = optarg;
      break;
    case 'r':
      config.dstGraph = optarg;
      break;
    case 'd':
      config.gpus = parseDevice(optarg), config.deviceId = config.gpus[0];
      break;
    case 'm':
      config.mt = parseMainTask(optarg);
      break;
    case 'p':
      config.level = parseLevel(optarg);
      break;
    case 'i':
      config.induced = parseInduced(optarg);
      break;
    case 'w':
      config.workerlist = parseWorkerList(optarg);
      break;
    case 'h':
      usage();
      exit(0);
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }

  if (config.mt == MAINTASK::MAINTASK_UNKNOWN)
  {
    fprintf(stderr, "Must specify -m option (Main Task)\n");
    usage();
    exit(0);
  }
  else if (config.mt == MAINTASK::CONVERT)
  {
    if (config.srcGraph == "")
    {
      fprintf(stderr, "Must specify -g option (Src graph FileName)\n");
      exit(0);
    }
    if (config.dstGraph == "")
    {
      fprintf(stderr, "Must specify -r option (Dst graph FileName)\n");
      exit(0);
    }
  }
  else
  {
    if (config.srcGraph == "")
    {
      fprintf(stderr, "Must specify -g option (Src graph FileName)\n");
      exit(0);
    }
    if (config.gpus.size() == 0)
    {
      fprintf(stderr, "Must specify -d option (Device Ids)\n");
      exit(0);
    }
    if (config.level == PARLEVEL::PARLEVEL_UNKNOWN)
    {
      fprintf(stderr, "Must specify -p option (Parallelization Scheme)\n");
      exit(0);
    }
    if (config.induced == INDUCEDSUBGRAPH::INDUCEDSUBGRAPH_UNKNOWN)
    {
      fprintf(stderr, "Must specify -i option (Induced Subgraphs Scheme)\n");
      exit(0);
    }
    if (config.workerlist == WORKERLIST::WORKERLIST_UNKNOWN)
    {
      fprintf(stderr, "Must specify -w option (Worker List Scheme)\n");
      exit(0);
    }
    if (config.mt == MAINTASK::MCE_BD_EVAL && config.workerlist == WORKERLIST::NOWL)
    {
      fprintf(stderr, "No breakdown for kernels without worker list\n");
      exit(0);
    }
    if (config.mt == MAINTASK::MCE_DONOR_EVAL && config.workerlist == WORKERLIST::NOWL)
    {
      fprintf(stderr, "No donoation for kernels without worker list\n");
      exit(0);
    }
    if (config.mt != MAINTASK::MCE && 1 != config.gpus.size())
    {
      fprintf(stderr, "Must use only one GPU to evaluate load balance, time breakdown or donation\n");
      exit(0);
    }
  }

  return config;
}

static void printConfig(Config config)
{
  printf("    Main Task = %s\n", asString(config.mt).c_str());
  printf("    Source Graph = %s\n", config.srcGraph.c_str());
  if (config.mt == MAINTASK::CONVERT)
  {
    printf("    Destination Graph = %s\n", config.dstGraph.c_str());
  }
  else
  {
    printf("    Parallelization Scheme = %s trees\n", asString(config.level).c_str());
    printf("    Induced Subgraphs Scheme = %s\n", asString(config.induced).c_str());
    printf("    Worker List Scheme = %s\n", asString(config.workerlist).c_str());
    printf("    Device ID(s) = %s\n", asString(config.gpus).c_str());
  }
  printf("-----------------------------\n");
}
