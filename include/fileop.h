#pragma once
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <sys/stat.h>
#include <vector>
#include "defs.h"
#include "logger.h"
using namespace std;

namespace graph
{
  template <typename T>
  void convert_to_bel(const string &path, const string &outputPath)
  {
    FILE *fp_ = fopen(path.c_str(), "r");
    vector<EdgeTy<T>> fileEdges;
    uint64 numread, src, dst, maxnode = 0;
    char line[1000];
    while ((fscanf(fp_, "%[^\n]%*c", line)) != EOF)
    {
      if (!isdigit(line[0]))
      {
        continue;
      }
      else
      {
        numread = sscanf(line, "%llu %llu", &src, &dst);

        if (numread == 2 && src != dst)
        {
          fileEdges.push_back(make_pair(T(src), T(dst)));
          fileEdges.push_back(make_pair(T(dst), T(src)));
          maxnode = max({maxnode, src, dst});
        }
      }
    }

    sort(fileEdges.begin(), fileEdges.end(), [](const EdgeTy<T> &a, const EdgeTy<T> &b) -> bool
         { return a.first < b.first || (a.first == b.first && a.second < b.second); });
    fileEdges.resize(unique(fileEdges.begin(), fileEdges.end()) - fileEdges.begin());

    uint64 m = fileEdges.size();

    FILE *writer = fopen(outputPath.c_str(), "wb");
    if (writer == nullptr)
    {
      return;
    }

    const T shard = 100000;
    uint64 *l;
    l = (uint64 *)malloc(3 * shard * sizeof(uint64));

    uint64 blocks = (m + shard - 1) / shard;
    for (uint64 i = 0; i < blocks; i++)
    {
      uint64 startEdge = i * shard;
      uint64 endEdge = (i + 1) * shard < m ? (i + 1) * shard : m;

      uint64 elementCounter = 0;
      for (uint64 j = startEdge; j < endEdge; j++)
      {
        EdgeTy<T> p = fileEdges[j];

        l[elementCounter++] = p.second;
        l[elementCounter++] = p.first;
        l[elementCounter++] = 0;
      }

      const uint64 numWritten = fwrite(l, 8, 3 * (endEdge - startEdge), writer);
    }

    fclose(writer);
    fclose(fp_);
    free(l);

    printf("n = %llu and m = %llu\n", maxnode + 1, m / 2);
  }

  template <typename T>
  void read_bel(const string &path, vector<EdgeTy<T>> &edges)
  {
    FILE *fp_ = fopen(path.c_str(), "r");
    if (nullptr == fp_)
    {
      Log(LogPriorityEnum::error, "unable to open %s", path.c_str());
    }

    const T shard = 100000;
    vector<char> belBuf_(24 * shard);
    size_t numRead = shard;

    struct stat st;
    stat(path.c_str(), &st);
    T m = st.st_size / 24, ptr = 0;
    edges.resize(m);

    do
    {
      numRead = fread(belBuf_.data(), 24, shard, fp_);
      if (numRead != shard)
      {
        if (feof(fp_))
        {
        }
        else if (ferror(fp_))
        {
          Log(LogPriorityEnum::error, "Error while reading %s: %s", path.c_str(), strerror(errno));
        }
        else
        {
          Log(LogPriorityEnum::error, "Unexpected error while reading %s", path.c_str());
        }
      }

      for (T i = 0; i < numRead; ++i)
      {
        uint64 src, dst;
        memcpy(&src, &belBuf_[i * 24 + 8], 8);
        memcpy(&dst, &belBuf_[i * 24 + 0], 8);
        edges[ptr].first = src;
        edges[ptr].second = dst;
        ++ptr;
      }
    } while (numRead == shard);

    fclose(fp_);
  }
} // namespace graph