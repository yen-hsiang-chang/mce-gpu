import os
import argparse
import urllib.request
import subprocess

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--graph", required = True, 
    help = "graph to download", 
    choices = ['wiki-talk', 'as-skitter', 'socfb-B-anon', 'soc-pokec', 'wiki-topcats', 'soc-livejournal',
    'soc-orkut', 'soc-sinaweibo', 'aff-orkut', 'clueweb09-50m', 'wiki-link', 'soc-friendster', 'all'])
  parser.add_argument("--dir", required = True, help = "path to the directory to store the graph")
  parser.add_argument('--force', action='store_true')

  return parser.parse_args()

args = get_args()

raw_dir = os.path.join(args.dir, 'mce_raw')

def download(cg):
  if os.path.exists(os.path.join(args.dir, cg + '.bel')) == True and args.force == False:
    print('The file {} already exists. Use the --force option to overwrite if needed.'.format(os.path.join(args.dir, cg + '.bel')))
    return
  def process_snap(fn):
    subprocess.run(['gzip', '-d', os.path.join(raw_dir, fn + '.txt.gz')])
    subprocess.run(['./parallel_mce_on_gpus', '-m', 'convert', '-g', os.path.join(raw_dir, fn + '.txt'), '-r', os.path.join(args.dir, cg + '.bel')])
    subprocess.run(['rm', '-r', raw_dir])

  def process_nrvis(fn):
    subprocess.run(['unzip', os.path.join(raw_dir, fn + '.zip'), '-d', raw_dir])
    if cg in ['soc-orkut', 'aff-orkut', 'clueweb09-50m', 'wiki-link']:
      subprocess.run(['./parallel_mce_on_gpus', '-m', 'convert', '-g', os.path.join(raw_dir, fn + '.edges'), '-r', os.path.join(args.dir, cg + '.bel')])
    else:
      subprocess.run(['./parallel_mce_on_gpus', '-m', 'convert', '-g', os.path.join(raw_dir, fn + '.mtx'), '-r', os.path.join(args.dir, cg + '.bel')])
    subprocess.run(['rm', '-r', raw_dir])

  if cg == 'wiki-talk':
    os.makedirs(raw_dir, exist_ok = True)
    urllib.request.urlretrieve('https://snap.stanford.edu/data/wiki-Talk.txt.gz', os.path.join(raw_dir, 'wiki-talk.txt.gz'))
    process_snap('wiki-talk')

  if cg == 'as-skitter':
    os.makedirs(raw_dir, exist_ok = True)
    urllib.request.urlretrieve('https://snap.stanford.edu/data/as-skitter.txt.gz', os.path.join(raw_dir, 'as-skitter.txt.gz'))
    process_snap('as-skitter')

  if cg == 'socfb-B-anon':
    os.makedirs(raw_dir, exist_ok = True)
    urllib.request.urlretrieve('https://nrvis.com/download/data/socfb/socfb-B-anon.zip', os.path.join(raw_dir, 'socfb-B-anon.zip'))
    process_nrvis('socfb-B-anon')

  if cg == 'soc-pokec':
    os.makedirs(raw_dir, exist_ok = True)
    urllib.request.urlretrieve('https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz', os.path.join(raw_dir, 'soc-pokec-relationships.txt.gz'))
    process_snap('soc-pokec-relationships')

  if cg == 'wiki-topcats':
    os.makedirs(raw_dir, exist_ok = True)
    urllib.request.urlretrieve('https://snap.stanford.edu/data/wiki-topcats.txt.gz', os.path.join(raw_dir, 'wiki-topcats.txt.gz'))
    process_snap('wiki-topcats')

  if cg == 'soc-livejournal':
    os.makedirs(raw_dir, exist_ok = True)
    urllib.request.urlretrieve('http://nrvis.com/download/data/soc/soc-livejournal.zip', os.path.join(raw_dir, 'soc-livejournal.zip'))
    process_nrvis('soc-livejournal')

  if cg == 'soc-orkut':
    os.makedirs(raw_dir, exist_ok = True)
    urllib.request.urlretrieve('https://nrvis.com/download/data/soc/soc-orkut-dir.zip', os.path.join(raw_dir, 'soc-orkut-dir.zip'))
    process_nrvis('soc-orkut-dir')
  
  if cg == 'soc-sinaweibo':
    os.makedirs(raw_dir, exist_ok = True)
    urllib.request.urlretrieve('https://nrvis.com/download/data/soc/soc-sinaweibo.zip', os.path.join(raw_dir, 'soc-sinaweibo.zip'))
    process_nrvis('soc-sinaweibo')
  
  if cg == 'aff-orkut':
    os.makedirs(raw_dir, exist_ok = True)
    urllib.request.urlretrieve('https://nrvis.com/download/data/massive/aff-orkut-user2groups.zip', os.path.join(raw_dir, 'aff-orkut-user2groups.zip'))
    process_nrvis('aff-orkut-user2groups')
  
  if cg == 'clueweb09-50m':
    os.makedirs(raw_dir, exist_ok = True)
    urllib.request.urlretrieve('https://nrvis.com/download/data/massive/web-ClueWeb09-50m.zip', os.path.join(raw_dir, 'web-ClueWeb09-50m.zip'))
    process_nrvis('web-ClueWeb09-50m')

  if cg == 'wiki-link':
    os.makedirs(raw_dir, exist_ok = True)
    urllib.request.urlretrieve('https://nrvis.com/download/data/massive/web-wikipedia_link_en13-all.zip', os.path.join(raw_dir, 'web-wikipedia_link_en13-all.zip'))
    process_nrvis('web-wikipedia_link_en13-all')
  
  if cg == 'soc-friendster':
    os.makedirs(raw_dir, exist_ok = True)
    urllib.request.urlretrieve('http://nrvis.com/download/data/massive/soc-friendster.zip', os.path.join(raw_dir, 'soc-friendster.zip'))
    process_nrvis('soc-friendster')

if args.graph == 'all':
  for cg in ['wiki-talk', 'as-skitter', 'socfb-B-anon', 'soc-pokec', 'wiki-topcats', 'soc-livejournal',
    'soc-orkut', 'soc-sinaweibo', 'aff-orkut', 'clueweb09-50m', 'wiki-link', 'soc-friendster']:
    download(cg)
else:
  download(args.graph)