import os
import argparse
import urllib.request
import subprocess
import re
import numpy as np
from tabulate import tabulate

inf = 1e100
graph_set = ['wiki-talk', 'as-skitter', 'socfb-B-anon', 'soc-pokec', 'wiki-topcats', 'soc-livejournal', 
             'soc-orkut', 'soc-sinaweibo', 'aff-orkut', 'clueweb09-50m', 'wiki-link', 'soc-friendster']
ratio = ['763.58', '319.41', '69.14', '316.04', '2407.49', '12.45', '131.67', '1442.95', '675.73', '1606.65', '3813.70', '17.15']
baseline_time = [4, 3, 2, 1, 2, 5, 93, 54, 147, 90, 109, 380]

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--output_dir", required = True, help = "path to the directory to store outputs")
  parser.add_argument("--task", required = True, 
    help = "task to run (mce: GPU best time, mce-donor-eval: number of donations, mce-opts-eval: GPU time for optimization combinations)",
    choices = ['mce', 'mce-donor-eval', 'mce-heuristics-eval'])

  return parser.parse_args()

args = get_args()

def parse_mce(fp):
  if os.path.exists(fp) == False:
    return inf
  with open(fp, 'r') as f:
    content = f.read()
    degentime = re.findall(r'Degeneracy ordering time: (.*) s', content)
    if len(degentime) != 1:
      return inf
    degentime = float(degentime[0])
    counter = re.findall(r'Found (.*) maximal cliques in total.', content)
    if len(counter) != 1:
      return inf
    counter = int(counter[0].replace(',', ''))
    if counter == 0:
      return inf
    timer = re.findall(r'count time (.*) s', content)
    if len(timer) != 1:
      return inf
    timer = float(timer[0])
    return timer + degentime

def parse_mce_donor(fp):
  if os.path.exists(fp) == False:
    return inf, ''
  with open(fp, 'r') as f:
    content = f.read()
    counter = re.findall(r'Found (.*) maximal cliques in total.', content)
    if len(counter) != 1:
      return inf, ''
    counter = int(counter[0].replace(',', ''))
    if counter == 0:
      return inf, ''
    ls = re.findall(r'([0-9]+), ([0-9]+), ([0-9]+)', content)
    total = 0
    for tup in ls:
      total += int(tup[2])
    timer = re.findall(r'count time (.*) s', content)
    if len(timer) != 1:
      return inf, ''
    timer = float(timer[0])
    return timer, str(total)

def parse_mce_heuristics(fp):
  if os.path.exists(fp) == False:
    return inf
  with open(fp, 'r') as f:
    content = f.read()
    counter = re.findall(r'Found (.*) maximal cliques in total.', content)
    if len(counter) != 1:
      return inf
    counter = int(counter[0].replace(',', ''))
    if counter == 0:
      return inf
    timer = re.findall(r'count time (.*) s', content)
    if len(timer) != 1:
      return inf
    timer = float(timer[0])
    return timer

def table_mce(odir):
  our_time = []
  for g in graph_set:
    cur_time = []
    for p in ['l1', 'l2']:
      for i in ['p', 'px']:
        for w in ['nowl', 'wl']:
          fp = os.path.join(args.output_dir, 'mce', g, p + '-i' + i + '-' + w + '-' + '1' + 'gpu.txt')
          cur_time.append(parse_mce(fp))
    best = min(cur_time)
    our_time.append(best)
  our_time = ['' if t == inf else '{:>.2f}'.format(t) for t in our_time]
  with open(os.path.join(odir, 'time.txt'), 'w') as f:
    print(tabulate({'Graph': graph_set, 'State-of-the-art CPU time (s)': baseline_time, 'GPU time (s)': our_time}, headers = 'keys', tablefmt = "grid", colalign = ('left', 'right', 'right')), file = f)

def table_mce_donor(odir):
  donations = {'l1' : [], 'l2' : []}
  for g in graph_set:
    for p in ['l1', 'l2']:
      best = inf, ''
      for i in ['p', 'px']:
        for w in ['wl']:
          fp = os.path.join(args.output_dir, 'mce-donor-eval', g, p + '-i' + i + '-' + w + '-' + '1' + 'gpu.txt')
          cur = parse_mce_donor(fp)
          if cur[0] < best[0]:
            best = cur
      donations[p].append(best[1])
  with open(os.path.join(odir, 'donation.txt'), 'w') as f:
    print(tabulate({'Graph': graph_set, 'L1': donations['l1'], 'L2': donations['l2']}, 
                     headers = 'keys', tablefmt = "grid", colalign = ('left', 'right', 'right')), file = f)

def table_mce_heuristics(odir):
  our_time = {'l1p' : [], 'l1px' : [], 'l2p' : [], 'l2px' : [], 'slowdown' : []}
  for g in graph_set:
    for p in ['l1', 'l2']:
      for i in ['p', 'px']:
        for w in ['wl']:
          fp = os.path.join(args.output_dir, 'mce', g, p + '-i' + i + '-' + w + '-' + '1' + 'gpu.txt')
          cur_time = parse_mce_heuristics(fp)
          our_time[p + i].append('' if cur_time == inf else '{:>.2f}'.format(cur_time))
  
  for idx in range(len(graph_set)):
    heur = our_time['l1p'][idx] if float(ratio[idx]) > 200 else our_time['l1px'][idx]
    if heur == '':
      our_time['slowdown'].append('')
    else:
      best = inf
      for p in ['l1', 'l2']:
        for i in ['p', 'px']:
          best = min(best, float(our_time[p + i][idx]) if our_time[p + i][idx] != '' else inf)
      our_time['slowdown'].append('{:>.2f}'.format(float(heur) / best))
  
  with open(os.path.join(odir, 'heuristics.txt'), 'w') as f:
    print(tabulate({'Graph': graph_set, '\u0394 / d' : ratio,
                    'L1 + IP': our_time['l1p'], 'L1 + IPX': our_time['l1px'], 
                    'L2 + IP': our_time['l2p'], 'L2 + IPX': our_time['l2px'],
                    'Heuristic slowdown' : our_time['slowdown']}, 
                    headers = 'keys', tablefmt = "grid", colalign = ('left', 'right', 'right', 'right', 'right', 'right', 'right')), file = f)

def table():
  if args.task == 'mce':
    odir = os.path.join(args.output_dir, 'table')
    os.makedirs(odir, exist_ok = True)
    table_mce(odir)
  if args.task == 'mce-donor-eval':
    odir = os.path.join(args.output_dir, 'table')
    os.makedirs(odir, exist_ok = True)
    table_mce_donor(odir)
  if args.task == 'mce-heuristics-eval':
    odir = os.path.join(args.output_dir, 'table')
    os.makedirs(odir, exist_ok = True)
    table_mce_heuristics(odir)

table()