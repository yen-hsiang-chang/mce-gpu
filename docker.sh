#!/bin/bash

if [ $# -ne 2 ]; then
    echo "[error]: number of arguments is incorrect."
    echo "  Usage: $0 /path/to/data/ /path/to/results/"
    echo "    /path/to/data/: the path to the directory on host that will be used as the place to store the graphs, mapping to /data/ inside the container"
    echo "    /path/to/results/: the path to the directory on host that will be used as the place to store the results, mapping to /results/ inside the container"
    exit 1
fi

data_dir=`realpath $1`

results_dir=`realpath $2`

if [[ "$(docker images -q mce:latest 2> /dev/null)" == "" ]]; then
    docker build . -t mce:latest
fi

echo "" && echo "Mapping $data_dir on host to /data in the container."

echo "Mapping $results_dir on host to /results in the container." && echo ""

echo "Launching docker container." && echo ""

docker run --rm --gpus=all -ti -v $data_dir:/data -v $results_dir:/results mce:latest

echo "" && echo "Please check $results_dir/plot and $results_dir/table for the evaluation results."