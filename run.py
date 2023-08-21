import os
import argparse
import urllib.request
import subprocess
import re
import numpy as np

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--graph", required = True, 
    help = "graph to evaluate", 
    choices = ['wiki-talk', 'as-skitter', 'socfb-B-anon', 'soc-pokec', 'wiki-topcats', 'soc-livejournal',
    'soc-orkut', 'soc-sinaweibo', 'aff-orkut', 'clueweb09-50m', 'wiki-link', 'soc-friendster', 'all'])
  parser.add_argument("--input_dir", required = True, help = "path to the directory storing the graph")
  parser.add_argument("--output_dir", required = True, help = "path to the directory to store outputs")
  parser.add_argument("--task", required = True, 
    help = "task to run (mce: normal run, mce-lb-eval: load balance evaluation, mce-bd-eval: time breakdown evaluation, mce-donor-eval: donation evaluation>)",
    choices = ['mce', 'mce-lb-eval', 'mce-bd-eval', 'mce-donor-eval'])
  parser.add_argument("--devices", required = True, 
    help = "device ID(s) to run experiments, separated by commas without spaces")

  return parser.parse_args()

args = get_args()
print(args)
exe = './parallel_mce_on_gpus'

def parse_scheme(content):
  ps = re.findall(r'Parallelization Scheme = (.*)', content)
  assert(len(ps) == 1)
  inds = re.findall(r'Induced Subgraphs Scheme = (.*)', content)
  assert(len(inds) == 1)

  ws = re.findall(r'Worker List Scheme = (.*)', content)
  assert(len(ws) == 1)
  return ps[0] + ' + ' + inds[0] + ' ' * (3 - len(inds[0])) + ' + ' + ws[0]

def parse_mce(fp):
  with open(fp, 'r') as f:
    content = f.read()
    scheme = parse_scheme(content)
    degentime = re.findall(r'Degeneracy ordering time: (.*) s', content)
    if len(degentime) != 1:
      print('| {:32} | {:>15} | {:>15} | {:>12} |'.format(scheme, 'OOM', 'OOM', 'OOM'))
      return
    degentime = float(degentime[0])
    counter = re.findall(r'Found (.*) maximal cliques in total.', content)
    if len(counter) != 1:
      print('| {:32} | {:>15.3f} | {:>15} | {:>12} |'.format(scheme, degentime, 'OOM', 'OOM'))
      return
    counter = int(counter[0].replace(',', ''))
    if counter == 0:
      print('| {:32} | {:>15.3f} | {:>15} | {:>12} |'.format(scheme, degentime, 'OOM', 'OOM'))
      return
    timer = re.findall(r'count time (.*) s', content)
    if len(timer) != 1:
      print('| {:32} | {:>15.3f} | {:>15} | {:>12} |'.format(scheme, degentime, 'OOM', 'OOM'))
      return
    timer = float(timer[0])
    print('| {:32} | {:>15.3f} | {:>15.3f} | {:>12} |'.format(scheme, degentime, timer, counter))

def parse_mce_lb(fp):
  with open(fp, 'r') as f:
    content = f.read()
    scheme = parse_scheme(content)
    counter = re.findall(r'Found (.*) maximal cliques in total.', content)
    if len(counter) != 1:
      print('| {:32} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} |'.format('Scheme', 'OOM', 'OOM', 'OOM', 'OOM', 'OOM'))
      return
    counter = int(counter[0].replace(',', ''))
    if counter == 0:
      print('| {:32} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} |'.format('Scheme', 'OOM', 'OOM', 'OOM', 'OOM', 'OOM'))
      return
    ls = re.findall(r'([0-9]+), ([0-9]+), ([0-9]+)', content)
    max_per_sm = []
    for tup in ls:
      sm, load = int(tup[1]), int(tup[2])
      while(sm >= len(max_per_sm)):
        max_per_sm.append(0)
      max_per_sm[sm] = max(max_per_sm[sm], load)
    data = np.array(max_per_sm, dtype = float)
    data /= np.mean(data)
    print('| {:32} | {:>8.6f} | {:>8.6f} | {:>8.6f} | {:>8.6f} | {:>8.6f} |'.format(
      scheme, np.percentile(data, 0), np.percentile(data, 25), 
      np.percentile(data, 50), np.percentile(data, 75), np.percentile(data, 100)))

def parse_mce_donor(fp):
  with open(fp, 'r') as f:
    content = f.read()
    scheme = parse_scheme(content)
    counter = re.findall(r'Found (.*) maximal cliques in total.', content)
    if len(counter) != 1:
      print('| {:32} | {:>19} |'.format('Scheme', 'OOM'))
      return
    counter = int(counter[0].replace(',', ''))
    if counter == 0:
      print('| {:32} | {:>19} |'.format('Scheme', 'OOM'))
      return
    ls = re.findall(r'([0-9]+), ([0-9]+), ([0-9]+)', content)
    total = 0
    for tup in ls:
      total += int(tup[2])
    print('| {:32} | {:>19} |'.format(scheme, total))

def parse_mce_bd(fp):
  with open(fp, 'r') as f:
    content = f.read()
    scheme = parse_scheme(content)
    counter = re.findall(r'Found (.*) maximal cliques in total.', content)
    if len(counter) != 1:
      print('| {:32} | {:>26} | {:>11} | {:>16} | {:>14} | {:>18} | {:>26} | {:>5} |'.format(
        scheme, 'OOM', 'OOM', 'OOM', 'OOM', 
        'OOM', 'OOM', 'OOM'))
      return
    counter = int(counter[0].replace(',', ''))
    if counter == 0:
      print('| {:32} | {:>26} | {:>11} | {:>16} | {:>14} | {:>18} | {:>26} | {:>5} |'.format(
        scheme, 'OOM', 'OOM', 'OOM', 'OOM', 
        'OOM', 'OOM', 'OOM'))
      return
    ls = re.findall(r'([0-9]+)' + r', ([0-9]+)' * 6, content)
    total = [0 for _ in range(7)]
    for tup in ls:
      for i in range(7):
        total[i] += int(tup[i])
    data = np.array(total, dtype = float)
    data /= np.sum(data)
    print('| {:32} | {:>26.1%} | {:>11.1%} | {:>16.1%} | {:>14.1%} | {:>18.1%} | {:>26.1%} | {:>5.1%} |'.format(
        scheme, data[1], data[2], data[3], data[4], data[5], data[6], data[0]))

def do_mce(cg, odir):
  ig = os.path.join(args.input_dir, cg + '.bel')
  print()
  print('Graph: ' + cg)
  print('| {:32} | {:15} | {:15} | {:12} |'.format('Scheme', 'Degen Time (s)', 'Count Time (s)', 'MCE Count'))
  for p in ['l1', 'l2']:
    for i in ['p', 'px']:
      for w in ['nowl', 'wl']:
        fp = os.path.join(odir, p + '-i' + i + '-' + w + '-' + str(args.devices.count(',') + 1) + 'gpu.txt')
        with open(fp , 'w') as f:
          subprocess.run([exe, '-m', args.task, '-d', args.devices, '-p', p, '-i', i, '-w', w, '-g', ig], stdout = f)
        parse_mce(fp)
        print('', end = '', flush = True)

def do_mce_lb(cg, odir):
  ig = os.path.join(args.input_dir, cg + '.bel')
  print()
  print('Graph: ' + cg)
  print('| {:32} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} |'.format('Scheme', '0%', '25%', '50%', '75%', '100%'))
  for p in ['l1', 'l2']:
    for i in ['p', 'px']:
      for w in ['nowl', 'wl']:
        fp = os.path.join(odir, p + '-i' + i + '-' + w + '-' + str(args.devices.count(',') + 1) + 'gpu.txt')
        with open(fp , 'w') as f:
          subprocess.run([exe, '-m', args.task, '-d', args.devices, '-p', p, '-i', i, '-w', w, '-g', ig], stdout = f)
        parse_mce_lb(fp)
        print('', end = '', flush = True)

def do_mce_bd(cg, odir):
  ig = os.path.join(args.input_dir, cg + '.bel')
  print()
  print('Graph: ' + cg)
  print('| {:32} | {:>26} | {:>11} | {:>16} | {:>14} | {:>18} | {:>26} | {:>5} | '.format(
    'Scheme', 'Building Induced Subgraphs', 'Worker List', 'Selecting Pivots', 
    'Set Operations', 'Testing Maximality', 'Branching and Backtracking', 'Other'))
  for p in ['l1', 'l2']:
    for i in ['p', 'px']:
      for w in ['wl']:
        fp = os.path.join(odir, p + '-i' + i + '-' + w + '-' + str(args.devices.count(',') + 1) + 'gpu.txt')
        with open(fp , 'w') as f:
          subprocess.run([exe, '-m', args.task, '-d', args.devices, '-p', p, '-i', i, '-w', w, '-g', ig], stdout = f)
        parse_mce_bd(fp)
        print('', end = '', flush = True)

def do_mce_donor(cg, odir):
  ig = os.path.join(args.input_dir, cg + '.bel')
  print()
  print('Graph: ' + cg)
  print('| {:32} | {:>19} |'.format('Scheme', 'Number of Donations'))
  for p in ['l1', 'l2']:
    for i in ['p', 'px']:
      for w in ['wl']:
        fp = os.path.join(odir, p + '-i' + i + '-' + w + '-' + str(args.devices.count(',') + 1) + 'gpu.txt')
        with open(fp , 'w') as f:
          subprocess.run([exe, '-m', args.task, '-d', args.devices, '-p', p, '-i', i, '-w', w, '-g', ig], stdout = f)
        parse_mce_donor(fp)
        print('', end = '', flush = True)

def do_experiment(cg):
  if args.task == 'mce':
    odir = os.path.join(args.output_dir, 'mce', cg)
    os.makedirs(odir, exist_ok = True)
    do_mce(cg, odir)
  if args.task == 'mce-lb-eval':
    odir = os.path.join(args.output_dir, 'mce-lb-eval', cg)
    os.makedirs(odir, exist_ok = True)
    do_mce_lb(cg, odir)
  if args.task == 'mce-bd-eval':
    odir = os.path.join(args.output_dir, 'mce-bd-eval', cg)
    os.makedirs(odir, exist_ok = True)
    do_mce_bd(cg, odir)
  if args.task == 'mce-donor-eval':
    odir = os.path.join(args.output_dir, 'mce-donor-eval', cg)
    os.makedirs(odir, exist_ok = True)
    do_mce_donor(cg, odir)
  
if args.task != 'mce' and len(args.devices) != 1:
  raise RuntimeError('Only allow single gpu for load balance and time breakdown evaluation.')

if args.graph == 'all':
  for cg in ['wiki-talk', 'as-skitter', 'socfb-B-anon', 'soc-pokec', 'wiki-topcats', 'soc-livejournal',
    'soc-orkut', 'soc-sinaweibo', 'aff-orkut', 'clueweb09-50m', 'wiki-link', 'soc-friendster']:
    do_experiment(cg)
else:
  do_experiment(args.graph)