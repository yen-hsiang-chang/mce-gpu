#!/bin/bash

echo "Downloading data..."
python3 download.py --graph all --dir /data

echo "Running strong scaling experiment with 1 GPU..."
python3 run.py --graph all --input_dir /data --output_dir /results --task mce --devices 0

echo "Running strong scaling experiment with 2 GPUs..."
python3 run.py --graph all --input_dir /data --output_dir /results --task mce --devices 0,1

echo "Running strong scaling experiment with 4 GPUs..."
python3 run.py --graph all --input_dir /data --output_dir /results --task mce --devices 0,1,2,3

echo "Running load balance evaluation..."
python3 run.py --graph all --input_dir /data --output_dir /results --task mce-lb-eval --devices 0

echo "Running time breakdown evaluation..."
python3 run.py --graph all --input_dir /data --output_dir /results --task mce-bd-eval --devices 0

echo "Running donation evaluation..."
python3 run.py --graph all --input_dir /data --output_dir /results --task mce-donor-eval --devices 0

echo "Plotting figures..."
python3 plot.py --output_dir /results --task mce
python3 plot.py --output_dir /results --task mce-lb-eval
python3 plot.py --output_dir /results --task mce-multigpu
python3 plot.py --output_dir /results --task mce-bd-eval

echo "Formatting tables..."
python3 table.py --output_dir /results --task mce
python3 table.py --output_dir /results --task mce-donor-eval
python3 table.py --output_dir /results --task mce-heuristics-eval

chmod -R 777 /data /results