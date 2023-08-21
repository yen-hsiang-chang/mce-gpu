import os
import argparse
import urllib.request
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

inf = 1e100
baseline_time = [4, 3, 2, 1, 2, 5, 93, 54, 147, 90, 109, 380]
graph_set = ['wiki-talk', 'as-skitter', 'socfb-B-anon', 'soc-pokec', 'wiki-topcats', 'soc-livejournal', 
             'soc-orkut', 'soc-sinaweibo', 'aff-orkut', 'clueweb09-50m', 'wiki-link', 'soc-friendster']

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--output_dir", required = True, help = "path to the directory to store outputs")
  parser.add_argument("--task", required = True, 
    help = "task to run (mce: speedup over baseline, mce-lb-eval: load balance evaluation, mce-bd-eval: time breakdown evaluation, mce-multigpu: strong scalability)",
    choices = ['mce', 'mce-lb-eval', 'mce-bd-eval', 'mce-multigpu'])

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

def parse_mce_lb(fp):
  if os.path.exists(fp) == False:
    return np.array([])
  with open(fp, 'r') as f:
    content = f.read()
    counter = re.findall(r'Found (.*) maximal cliques in total.', content)
    if len(counter) != 1:
      return np.array([])
    counter = int(counter[0].replace(',', ''))
    if counter == 0:
      return np.array([])
    ls = re.findall(r'([0-9]+), ([0-9]+), ([0-9]+)', content)
    max_per_sm = []
    for tup in ls:
      sm, load = int(tup[1]), int(tup[2])
      while(sm >= len(max_per_sm)):
        max_per_sm.append(0)
      max_per_sm[sm] = max(max_per_sm[sm], load)
    data = np.array(max_per_sm, dtype = float)
    data /= np.mean(data)
    return data

def parse_mce_bd(fp):
  if os.path.exists(fp) == False:
    return np.zeros(7)
  with open(fp, 'r') as f:
    content = f.read()
    counter = re.findall(r'Found (.*) maximal cliques in total.', content)
    if len(counter) != 1:
      return np.zeros(7)
    counter = int(counter[0].replace(',', ''))
    if counter == 0:
      return np.zeros(7)
    ls = re.findall(r'([0-9]+)' + r', ([0-9]+)' * 6, content)
    total = [0 for _ in range(7)]
    for tup in ls:
      for i in range(7):
        total[i] += int(tup[i])
    data = np.array(total, dtype = float)
    data /= np.sum(data)
    return np.array([data[0], data[2], data[1], data[3], data[4], data[5], data[6]])

# In strong scalability, the degeneracy time is not included since it is not scaled to multi-GPUs
def parse_mce_multigpu(fp):
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

def plot_mce(odir):
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
  graph_set.append('')
  graph_set.append('Geomean')
  x = np.arange(len(graph_set))
  base = np.array(baseline_time) / np.array(baseline_time)
  base = np.append(base, [0, 1])
  our = np.array(baseline_time) / np.array(our_time)
  valid = np.array([v for v in our if v > 1e-50])
  geomean = 0
  if(len(valid) != 0):
    geomean = valid.prod()**(1.0/len(valid))
  our = np.append(our, [0, geomean])
  width = 0.3
  plt.figure(dpi = 300, figsize = (8, 7))
  plt.bar(x, base, width, color  = 'green', label = 'Skylake with 96 threads', edgecolor = 'black')
  plt.bar(x + width, our, width, color = 'blue', label = 'Single GPU', edgecolor = 'black')
  plt.xticks(x + width / 2, graph_set, rotation = 30, ha = 'right')
  plt.ylabel('Speedup (over Skylake with 96 threads)')
  plt.title('Speedup over the state-of-the-art parallel CPU implementation')
  plt.yscale('log', base = 2)
  ax = plt.gca()
  ax.get_yaxis().set_major_formatter(FormatStrFormatter('%.1f'))
  plt.ylim([0.5, 64])
  plt.legend()
  
  plt.savefig(os.path.join(odir, 'speedup.png'))

def plot_mce_lb(odir):
  dps = []
  for g in graph_set:
    dp = []
    for w in ['nowl', 'wl']:
      for p in ['l1', 'l2']:
        for i in ['p']:
          fp = os.path.join(args.output_dir, 'mce-lb-eval', g, p + '-i' + i + '-' + w + '-' + '1' + 'gpu.txt')
          dp.append(parse_mce_lb(fp))
    dps.append(dp)

  plt.rc('axes', titlesize = 18)
  plt.rc('axes', labelsize = 20)
  plt.rc('xtick', labelsize = 16)
  plt.rc('ytick', labelsize = 16)
  plt.rc('figure', titlesize = 24)
  fig, axes = plt.subplots(1, 12, sharey = True)
  fig.set_dpi(300)
  fig.set_size_inches(30, 8)
  fig.suptitle('Load distribution across streaming multiprocessors (SMs) with first- or second-level sub-trees and with or without using a worker list')
  plt.yscale('log', base = 2)
  ax = plt.gca()
  ax.get_yaxis().set_major_formatter(FormatStrFormatter('%.3f'))
  
  for i in range(12):
    box1 = axes[i].boxplot(dps[i], whis = inf, patch_artist=True)
    plt.setp(box1['medians'], color = 'black')
    plt.setp(box1['boxes'], facecolor = 'green')
    if len(dps[i][0]) != 0:
      box2 = axes[i].boxplot(np.clip(dps[i], a_min = np.zeros((len(dps[i]), len(dps[i][0]))), a_max = np.expand_dims(np.median(dps[i], axis = 1), 1).repeat(len(dps[i][0]), axis = 1)).tolist(), whis = inf, patch_artist=True)
      plt.setp(box2['medians'], color = 'black')
      plt.setp(box2['boxes'], facecolor = 'gold')
    
    axes[i].set_title( 'No WL       WL  \n' + graph_set[i], y = -0.15)
    axes[i].set_xticks(1 + np.arange(4), ['L1', 'L2', 'L1', 'L2'])
    if i == 0:
      axes[i].set_ylabel('Load distribution across SMs\n(normalized to average)')
      axes[i].set_ylim([0.125, 8])
      
  plt.tight_layout()
  plt.subplots_adjust(wspace = 0)
  plt.savefig(os.path.join(odir, 'load-balance.png'))
  

def plot_mce_bd(odir):
  our_time = []
  our_percentage = []
  for g in graph_set:
    cur_time = []
    cur_percentage = []
    for p in ['l1', 'l2']:
      for i in ['p', 'px']:
        for w in ['wl']:
          fp_time = os.path.join(args.output_dir, 'mce', g, p + '-i' + i + '-' + w + '-' + '1' + 'gpu.txt')
          cur_time.append(parse_mce(fp_time))
          fp = os.path.join(args.output_dir, 'mce-bd-eval', g, p + '-i' + i + '-' + w + '-' + '1' + 'gpu.txt')
          cur_percentage.append(parse_mce_bd(fp))
    our_time.append(cur_time)
    our_percentage.append(cur_percentage)

  plt.rc('axes', titlesize = 16)
  plt.rc('axes', labelsize = 16)
  plt.rc('xtick', labelsize = 16)
  plt.rc('ytick', labelsize = 16)
  plt.rc('legend', fontsize = 16)
  plt.rc('figure', titlesize = 24)
  fig, axes = plt.subplots(2, 6)
  fig.set_dpi(300)
  fig.set_size_inches(24, 10)
  fig.suptitle('Breakdown and comparison of execution time with first- or second-level sub-trees and with partial or full induced sub-graphs')
  labels = ['Other', 'Worker list operations', 'Building induced subgraphs', 'Selecting pivots', 'Set operations', 'Testing for maximality', 'Branching and backtracking']
  colors = ['black', 'red', 'gold', 'green', 'blue', 'cyan', 'magenta']
  for i in range(2):
    axes[i][0].set_ylabel('Execution Time (seconds)')
    for j in range(6):
      total = np.zeros(4)
      x = 1 + np.arange(4)
      for k in range(7):
        cur_h = np.array(our_percentage[i * 6 + j])[:, k] * np.array(our_time[i * 6 + j])
        axes[i][j].bar(x, cur_h, bottom = total, color = colors[k], label = labels[k], width = 0.5)
        total += cur_h
      axes[i][j].set_title( 'L1                  L2\n' + graph_set[i * 6 + j], y = -0.25)
      axes[i][j].set_xticks(1 + np.arange(4), ['IP', 'IPX', 'IP', 'IPX'])
      axes[i][j].get_yaxis().set_major_formatter(FormatStrFormatter('%.2f'))
      axes[i][j].set_ylim(bottom = 0)
      
  plt.tight_layout()
  plt.subplots_adjust(wspace = 0.3, hspace = 0.5)
  plt.legend(ncol = 7, bbox_to_anchor=(0.8, 1.25))
  plt.text(-1.075, 0.003, 'Out of Memory', rotation = 'vertical', fontsize = 16)
  plt.text(-3.075, 0.003, 'Out of Memory', rotation = 'vertical', fontsize = 16)
  plt.savefig(os.path.join(odir, 'breakdown.png'))

def plot_mce_multigpu(odir):
  our_time = []
  for g in graph_set:
    cur_time = []
    for w in ['wl', 'nowl']:
      for p in ['l1', 'l2']:
        for i in ['p']:
          gpu_time = []
          for c in ['1', '2', '4']:
            fp = os.path.join(args.output_dir, 'mce', g, p + '-i' + i + '-' + w + '-' + c + 'gpu.txt')
            gpu_time.append(parse_mce_multigpu(fp))
          cur_time.append(gpu_time)
    our_time.append(cur_time)

  labels = ['WL + L1', 'WL + L2', 'No WL + L1', 'No WL + L2']
  colors = ['green', 'green', 'gold', 'gold']
  markers = ['s', '^', 's', '^']
  mss = [15, 15, 12, 12]
  plt.rc('axes', titlesize = 18)
  plt.rc('axes', labelsize = 20)
  plt.rc('xtick', labelsize = 16)
  plt.rc('ytick', labelsize = 16)
  plt.rc('legend', fontsize = 16)
  plt.rc('figure', titlesize = 24)
  fig, axes = plt.subplots(1, 12, sharey = True)
  fig.set_dpi(300)
  fig.set_size_inches(30, 8)
  fig.suptitle('Strong scaling with respect to the number of GPUs with first- or second-level sub-trees and with or without using a worker list')
  plt.yscale('log', base = 2)
  ax = plt.gca()
  ax.get_yaxis().set_major_formatter(FormatStrFormatter('%.2f'))

  for i in range(12):
    base = our_time[i][2][0]
    for j in range(4):
      if(base != inf):
        axes[i].plot(np.arange(3) + 1, base / np.array(our_time[i][j]), label = labels[j], mec = 'black', mfc = colors[j], marker = markers[j], c = colors[j], lw = 3, ms = mss[j])
      else: # do no show
        axes[i].plot(np.arange(3) + 1, [inf, inf, inf], label = labels[j], mec = 'black', mfc = colors[j], marker = markers[j], c = colors[j], lw = 3, ms = mss[j])
    axes[i].set_title(graph_set[i], y = -0.1)
    axes[i].set_xticks(1 + np.arange(3), ['1', '2', '4'])
    axes[i].set_xlim([0.5, 3.5])
    if i == 0:
      axes[i].set_ylabel('Speedup\n(over No WL + L1 + 1GPU)')
      axes[i].set_ylim([0.25, 64])
      
  plt.tight_layout()
  plt.subplots_adjust(wspace = 0)
  plt.legend()
  plt.text(-34.1, 0.20, '# of GPUs = ', fontsize = 14)
  plt.savefig(os.path.join(odir, 'multigpu.png'))

def plot():
  if args.task == 'mce':
    odir = os.path.join(args.output_dir, 'plot')
    os.makedirs(odir, exist_ok = True)
    plot_mce(odir)
  if args.task == 'mce-multigpu':
    odir = os.path.join(args.output_dir, 'plot')
    os.makedirs(odir, exist_ok = True)
    plot_mce_multigpu(odir)
  if args.task == 'mce-lb-eval':
    odir = os.path.join(args.output_dir, 'plot')
    os.makedirs(odir, exist_ok = True)
    plot_mce_lb(odir)
  if args.task == 'mce-bd-eval':
    odir = os.path.join(args.output_dir, 'plot')
    os.makedirs(odir, exist_ok = True)
    plot_mce_bd(odir)

plot()