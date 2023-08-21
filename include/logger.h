#pragma once
#include "defs.h"

template <bool NEWLINE = true, typename... Args>
void Log(LogPriorityEnum l, const char *f, Args... args)
{
  // Line Color Set
  switch (l)
  {
  case critical:
  case error:
    printf("\033[1;31m");
    break;
  case info:
    printf("\033[1;32m");
    break;
  case warn:
    printf("\033[1;33m");
    break;
  case debug:
    printf("\033[1;34m");
    break;
  default:
    printf("\033[0m");
  }

  time_t rawtime;
  struct tm *timeinfo;
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  printf("[%02d:%02d:%02d] ", timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);

  printf(f, args...);

  if (NEWLINE)
    printf("\n");

  if (l == critical || l == error)
    exit(0);

  printf("\033[0m");
}