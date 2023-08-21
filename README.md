# Parallelizing Maximal Clique Enumeration on GPUs

This artifact aims for reproducing experiments mentioned in the "Parallelizing Maximal Clique Enumeration on GPUs" paper in PACT 2023 by generating key figures and tables. 

## Minimum Requirements

The following **minimum** requirements need to be satisfied in order to reproduce the experiments:

### Hardware
- A CPU with 4 cores in x86_64 architecture, 128 GB of RAM and 256 GB of disk space
- 4 NVIDIA GPUs with compute capability 7.0 or higher (i.e., Volta architecture or later) and with 32 GB of GPU memory each

### Software
- Operating system of Ubuntu 20.04 or CentOS 8
- CUDA Toolkit 11 with driver version of 450.80.02 to have built-in CUB library, or CUDA Toolkit 10.2 with driver version of 440.33 and with CUB library from source
- GCC 8 with OpenMP 4.5 to have a unique CPU thread for each GPUs
- Python 3.6 with numpy, matplotlib and tabulate to run the pre-compiled executable, plotting figures and formatting tables

## Running with Docker
To simplify the setup, we provide a Dockerfile that builds a docker image satisfying the software requirements stated above. Since GPUs are required in the experiements, please follow the [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to make sure that GPUs are ready to be used with Docker.

To start with, the [docker.sh](docker.sh) script is provided to build the docker image and run the docker container with the places to store graphs and results configured. The usage is as follows:

```
Usage: ./docker.sh /path/to/data/ /path/to/results/
  /path/to/data/: the path to the directory on host that will be used as the place to store the graphs, mapping to /data/ inside the container
  /path/to/results/: the path to the directory on host that will be used as the place to store the results, mapping to /results/ inside the container
```

For example, running the following command on host builds the docker image and run the docker container with `/path/to/data/` on host mapping to `/data/` inside the container, and with `/path/to/results/` on host mapping to `/results/` inside the container. Note that `/path/to/data/` needs to have the ability to store all input graphs, which is about 200 GB. 

```
./docker.sh /path/to/data/ /path/to/results/
```

Once the docker container is launched, we can reproduce the experiments inside the container. Although several scripts are provided to reproduce different experiments (with their usages descrbied later for completeness), one integrated script, [all_experiments.sh](all_experiments.sh), is provided for convenience. Running the following command inside the container downloads the graphs into `/data/` (`/path/to/data/` on host), runs through all experiements and stores the raw results, generated figures and tables into `/results/` (`/path/to/results/` on host). The whole experiments take about six hours, with some fluctuations since downloading datasets depends on the internet bandwidth.

```
./all_experiments.sh
```

After the experiments are done, we can exit the docker container and inspect the results in `/path/to/results/` on host. We do not expect major differences between running inside the container and running in the local environment. However, we expect some minor differences for the GPU time reported in Figure 5 and Table 4 in the paper, as optimization combinations are sensitive to memory bandwidth and computing power on GPUs, and different GPUs have different characteristics.

## Expected Results

After the experiments are done, results are stored in `/path/to/results/` on host. Each of figures in `/path/to/results/plot/` corresponds to a figure in the evaluation section of the paper, and each of tables in `/path/to/results/table/` corresponds to a table in the evaluation section of the paper. The descriptions are as follows and please refer to the paper for more details:

### GPU Time and Speedup
The table `/path/to/results/table/time.txt` outputs the GPU time reported in Table 1. Note that the time here includes both the degeneracy ordering time and the maximal clique counting time.

### Load Distribution and Worker List Utilization
The figure `/path/to/results/plot/load-balance.png` corresponds to Figure 3 and the table `/path/to/results/table/donation.txt` outputs the number of donations reported in Table 3 in the paper.

### Strong Scalability
The figure `/path/to/results/plot/multigpu.png` corresponds to Figure 4 in the paper, which evaluates strong scalability with 1, 2 and 4 GPUs.

### Time Breakdown and Heuristics of Optimization Combinations
The figure `/path/to/results/plot/breakdown.png` corresponds to Figure 5 and the table `/path/to/results/table/heuristics.txt` outputs the GPU time reported in Table 4 in the paper. Note that in this set of experiments we exclude the time spent on preprocessing degeneracy ordering as optimization combinations only affect the kernel times on counting maximal cliques. Therefore, the times reported here are not the same as those reported in the GPU time and speedup section. We expect some minor differences for the GPU time reported here, as optimization combinations are sensitive to memory bandwidth and computing power on GPUs, and different GPUs have different characteristics.

At this point, key tables and figures in the paper generated by this artifact have been reproduced. The following sections are for completeness, providing the workflow of experiments executed in [all_experiments.sh](all_experiments.sh) and usages for each of the scripts. 

## Workflow
This section describes the workflow of experiments executed in [all_experiments.sh](all_experiments.sh), its relationships with other components and usages of each of the scripts. The whole workflow is expected to take about six hours.
### Pre-compiled Executable
All experiments are reproduced on top of the pre-compiled executable, [parallel_mce_on_gpus](parallel_mce_on_gpus), with its usage as follows:
```
Usage:  ./parallel_mce_on_gpus [options]

Options:
    -g <Src graph FileName>       Name of file with source graph
    -r <Dst graph FileName>       Name of file with destination graph only for conversion
    -d <Device Id(s)>             GPU Device Id(s) separated by commas without spaces
    -m <Main Task>                Name of the task to perform <convert: graph conversion, mce, mce-lb-eval: load balance evaluation, mce-bd-eval: breakdown evaluation, mce-donor-eval: donation evaluation>
    -p <Parallelization Scheme>   Level of subtrees to parallelize <l1: first level, l2: second level>
    -i <Induced Subgraphs Scheme> Building induced subgraphs from which sets <p: P only, px: P and X>
    -w <Worker List Scheme>       Use worker list to achieve load balance or not <nowl: No worker list, wl: Use worker list>
    -h                            Help
```
To facilitate reproducing experiments, several scripts are provided with their usage described below to scan through different combinations of schemes and experiments.

### Preparing Datasets
In [all_experiments.sh](all_experiments.sh), we first run the following command to prepare datasets for evaluation:
```
# This takes about 1 hour.
python3 download.py --graph all --dir /data
```

The graphs used for evaluation are from the [SNAP Datasets](https://snap.stanford.edu/) and the [Network Repository](https://networkrepository.com), as shown in the table below:

| Graph | # of nodes | # of edges | Max degree | Degeneracy | # of maximal cliques | 
| :---- | ---------: | ---------: | ---------: | ---------: | -------------------: |
| [wiki-talk](https://snap.stanford.edu/data/wiki-Talk.html)                 | 2,394,385   | 4,659,565     | 100,029    | 131        | 86,333,306    |
| [as-skitter](https://snap.stanford.edu/data/as-Skitter.html)               | 1,696,415   | 11,095,298    | 35,455     | 111        | 37,322,355    | 
| [socfb-B-anon](https://networkrepository.com/socfb-B-anon.php)             | 2,937,613   | 20,959,854    | 4,356      | 63         | 27,593,398    |
| [soc-pokec](https://snap.stanford.edu/data/soc-Pokec.html)                 | 1,632,804   | 22,301,964    | 14,854     | 47         | 19,376,873    |
| [wiki-topcats](https://snap.stanford.edu/data/wiki-topcats.html)           | 1,791,489   | 25,444,207    | 238,342    | 99         | 27,229,873    | 
| [soc-livejournal](https://networkrepository.com/soc-livejournal.php)       | 4,033,138   | 27,933,062    | 2,651      | 213        | 38,413,665    |
| [soc-orkut](https://networkrepository.com/soc-orkut-dir.php)               | 3,072,442   | 117,185,083   | 33,313     | 253        | 2,269,631,973 |
| [soc-sinaweibo](https://networkrepository.com/soc-sinaweibo.php)           | 58,655,850  | 261,321,033   | 278,489    | 193        | 1,117,416,174 | 
| [aff-orkut](https://networkrepository.com/aff-orkut-user2groups.php)       | 8,730,858   | 327,036,486   | 318,268    | 471        | 417,032,363   |
| [clueweb09-50m](https://networkrepository.com/web-ClueWeb09-50m.php)       | 428,136,613 | 446,766,953   | 308,477    | 192        | 1,001,323,679 |
| [wiki-link](https://networkrepository.com/web-wikipedia-link-en13-all.php) | 27,154,799  | 543,183,611   | 4,271,341  | 1,120      | 568,730,123   | 
| [soc-friendster](https://networkrepository.com/soc-friendster.php)         | 65,608,367  | 1,806,067,135 | 5,214      | 304        | 3,364,773,700 |

For completeness, the usage of the script, [download.py](download.py), is provided as follows:
```
python3 download.py --dir DIR --graph GRAPH
```
The argument descriptions are as follows:

- `--dir DIR`: The path to the directory to store the graph.
- `--graph GRAPH`: The graph to download, selected from `{wiki-talk,as-skitter,socfb-B-anon,soc-pokec,wiki-topcats,soc-livejournal,soc-orkut,soc-sinaweibo,aff-orkut,clueweb09-50m,wiki-link,soc-friendster,all}`. To download graphs all at once, please set this argument to `all`.

After the graph is downloaded, it is converted as `DIR/GRAPH.bel`. Note that to store all 12 graphs used for evaluation, 200 GB of disk space is needed.

### Running Experiments
In [all_experiments.sh](all_experiments.sh), after preparing the datasets, we utilize the [run.py](run.py) script to run experiments and generate raw results going to be used for plotting figures and formatting tables. Note that the [run.py](run.py) script relies on the pre-compiled executable.

Basic GPU time and strong scalability experiments, capturing the degeneracy ordering time and maximal clique counting time on different number of GPUs:
```
# These take about 3 hours.
python3 run.py --graph all --input_dir /data --output_dir /results --task mce --devices 0
python3 run.py --graph all --input_dir /data --output_dir /results --task mce --devices 0,1
python3 run.py --graph all --input_dir /data --output_dir /results --task mce --devices 0,1,2,3
```
Load balance experiments, counting the number of traversed nodes for each blocks on single GPU:
``` 
# This takes about 1 hour.
python3 run.py --graph all --input_dir /data --output_dir /results --task mce-lb-eval --devices 0
```
Time breakdown experiments, counting the number of cycles spent in different categories on single GPU:
```
# This takes about 0.5 hours.
python3 run.py --graph all --input_dir /data --output_dir /results --task mce-bd-eval --devices 0
```
Number of donations, counting the number of donating events among workers on single GPU:
```
# This takes about 0.5 hours.
python3 run.py --graph all --input_dir /data --output_dir /results --task mce-donor-eval --devices 0
```

For completeness, the usage of the script, [run.py](run.py), is provided as follows:
```
python3 run.py --task TASK --devices DEVICES --input_dir INPUT_DIR --output_dir OUTPUT_DIR --graph GRAPH
```
The argument descriptions are as follows:

- `--task TASK`: The task to run, selected from `{mce,mce-lb-eval,mce-bd-eval,mce-donor-eval}`.
    - `mce`: This task reports the number of maximal cliques in a graph and the latency for pre-processing and counting, excluding the time on loading the graph. This task can be run on single GPU or multiple GPUs.
    - `mce-lb-eval`: This task reports the load distribution across SMs normalized to average, represented as quartiles. The load of an SM is measured as the maximum number of tree nodes visited by any block on that SM. This task can only be run on single GPU.
    - `mce-bd-eval`: This task reports the breakdown of execution time into seven categories, including `{building induced subgraphs, worker list operations, set operations, selecting pivots, testing for maximality, branching and backtracking, other}`. This task can only be run on single GPU.
    - `mce-donor-eval`: This task reports the number of donations among workers when utilizing worker list. This task can only be run on single GPU.
- `--devices DEVICES`: The device ID(s) to run experiments, separated by commas without spaces. For example, use `--devices 3` to perform single-GPU evaluation on device 3 and use `--devices 0,1,2,3` to perform multi-GPU evaluation on devices 0 to 3.
- `--input_dir INPUT_DIR`: The path to the directory storing the graphs, which is the same as `DIR` in the downloading script.
- `--output_dir OUTPUT_DIR`: The path to the directory to store raw reports used for analysis.
- `--graph GRAPH`: The input graph, selected from `{wiki-talk,as-skitter,socfb-B-anon,soc-pokec,wiki-topcats,soc-livejournal,soc-orkut,soc-sinaweibo,aff-orkut,clueweb09-50m,wiki-link,soc-friendster,all}`. To evaluate graphs all at once, please set this argument to `all`. The input graph is supposed to be `INPUT_DIR/GRAPH.bel`, which is the place the downloading script stores the converted graph.

### Plotting Figures
In [all_experiments.sh](all_experiments.sh), after running experiments, we utilize the [plot.py](plot.py) script to plot figures from the generated raw data, by running the following commands:

Speedup over the state-of-the-art implementation:
```
python3 plot.py --output_dir /results --task mce
```
Load distribution across streaming multiprocessors:
```
python3 plot.py --output_dir /results --task mce-lb-eval
```
Strong scaling experiments with 1, 2 and 4 GPUs:
```
python3 plot.py --output_dir /results --task mce-multigpu
```
Breakdown of execution time with different optimization combinations:
```
python3 plot.py --output_dir /results --task mce-bd-eval
```
Each of the commands above generates a figure in `/results/plot/`.

For completeness, the usage of the script, [plot.py](plot.py), is provided as follows:
```
python3 plot.py --output_dir OUTPUT_DIR --task {mce,mce-lb-eval,mce-bd-eval,mce-multigpu}
```
The argument descriptions are as follows:

- `--output_dir OUTPUT_DIR`: The path to the directory storing raw reports, which is the same as the one specified when running [run.py](run.py). This directory is also used for storing figures.
- `--task TASK`: The figure to plot, selected from `{mce,mce-lb-eval,mce-bd-eval,mce-multigpu}`.
    - `mce`: This plots the figure of speedup over the state-of-the-art implementation and stores as `OUTPUT_DIR/plot/speedup.png`.
    - `mce-lb-eval`: This plots the figure of load distribution across streaming multiprocessors and stores as `OUTPUT_DIR/plot/load-balance.png`.
    - `mce-bd-eval`: This plots the figure of breakdown of execution time with different optimization combinations and stores as `OUTPUT_DIR/plot/breakdown.png`.
    - `mce-multigpu`: This plots the figure of strong scaling experiments with 1, 2 and 4 GPUs and stores as `OUTPUT_DIR/plot/multigpu.png`.

### Formatting Tables
In [all_experiments.sh](all_experiments.sh), along with plotting figures, we also format tables from the generated raw data using the [table.py](table.py) script, by running the following commands:

Best execution time on single GPU:
```
python3 table.py --output_dir /results --task mce
```
Number of donations among workers:
```
python3 table.py --output_dir /results --task mce-donor-eval
```
Execution time of different optimization combinations and the heuristic selection slowdown:
```
python3 table.py --output_dir /results --task mce-heuristics-eval
```

Each of the commands above generates a table in `/results/table/`.

For completeness, the usage of the script, [table.py](table.py), is provided as follows:
```
python3 table.py --output_dir OUTPUT_DIR --task {mce,mce-donor-eval,mce-heuristics-eval}
```
The argument descriptions are as follows:

- `--output_dir OUTPUT_DIR`: The path to the directory storing raw reports, which is the same as the one specified when running [run.py](run.py). This directory is also used for storing tables.
- `--task TASK`: The figure to plot, selected from `{mce,mce-donor-eval,mce-heuristics-eval}`.
    - `mce`: This generates the table of best execution times on single GPU and stores as `OUTPUT_DIR/table/time.txt`.
    - `mce-donor-eval`: This generates the number of donations among workers and stores as `OUTPUT_DIR/table/donation.txt`.
    - `mce-heuristics-eval`: This generates the execution time of different optimization combinations and the heuristic selection slowdown and stores as `OUTPUT_DIR/table/heuristics.txt`.
