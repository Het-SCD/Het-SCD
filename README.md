Het-SCD: High-Quality Community Detection on Heterogeneous Platforms
===
Het-SCD [1] is a heterogenous implementation of the SCD [2] community detection algorithm. It is designed for heterogenous platforms consisting of a multi-core CPU and one or more NVIDIA GPUs.


Compile
===
Het-SCD requires the following packages to be installed (in parentheses are the recommended versions):

* NVIDIA Cuda Compiler (`nvcc` 5.5)
* GNU Compiler Collection (`gcc` 4.8.2)
* GNU Make (3.81)
* [Modern GPU Library](https://github.com/NVlabs/moderngpu) by NVlabs
* [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) by Karypis Lab (5.1.0)

Download Modern GPU and METIS from the provided websites, compile them and modify `config.mk` in the root directory of Het-SCD to refer to the root directory of these libraries.

```
MGPU_PATH=<path to Modern GPU>
METIS_PATH=<path to METIS>
```

Next, run `make all` in the root directory to compile all source files.

```
make all
```

Usage
===
To use Het-SCD, one first needs to convert a network file from a human-readable format into a binary format using the `convert` program.

```
./convert [text graph file] [binary graph file]
```

This program reads the graph file, performs some preprocessing steps and stores it as a binary file. The graph file should contain one edge per line
where each edge consists of a pair of two numbers. Since the network is interpreted as an undirected network, the order of endpoints of an edge is
irrelevant (`1 2` is equivalent to `2 1`). Duplicate edges, empty lines or lines starting with a `#` are ignored. Here is an example of a valid
network file.

```
1 2
2 3
3 5
1 5
```

Het-SCD consists of three version of the SCD algorithm: a CPU-only version, a GPU-only version and a heterogenous version for a CPU and one or more GPUs. To use the CPU-only version, execute:

```
OMP_NUM_THREADS=[number of threads] ./main [binary graph file]
```

To use the GPU-only version, run:

```
./main-gpu [binary graph file] [device]
```

Where the second argument is the ordinal number of the device to use. Use `0` if only one GPU is available, otherwise use `nvidia-smi` to find the identifiers of the available GPUs.

To use the heterogenous version, one first needs to compute a partion of the graph using the `partition` program.

```
./partition [binary graph file] [partition file] [number of devices] [weight CPU] [weight GPU 1] [weight GPU 2] ... [weight GPU n]
```

This program reads the graph file, partitions the graph and store the result into the partition file. The number of devices is the number of CPUs and GPUs to use, currently it is always the number of GPUs plus one. The weights indicate the number of vertices to assign to each device. If device A has twice the weight of device B then device A will get twice the number of vertices compared to device B. Some examples:

```
# One CPU + one GPU: both devices get 50%
./partition example.graph example.part 2 0.5 0.5

# Two GPUs: CPU gets 0%, GPU 1 gets 25% and GPU 2 gets 75%
./partition example.graph example.part 3 0 0.25 0.75

# One GPU + two GPUS: CPU gets 80%, both GPUs get 10%
./partition example.graph example.part 3 0.8 0.1 0.1
```

Finally, run SCD using the following commmand:

```
./main-dist [binary graph file] [partition file] [device ids]
```

The third argument should be the ordinal numbers of the devices to use, seperated by commas. For example, `2,3,0` means GPU 1 be the GPU with ordinal number 2, GPU 2 will be the GPU with ordinal number 3 and so on.

Example of usage
===
Below is an example of how to use the programs explained in the previous sections.

```
# Convert graph
./convert example.txt example.graph

# Run on CPU
./main example.graph

# Run on GPU
./main-gpu example.graph 0

# Partition graph: 50% to CPU, 25% to GPUs
./partition example.graph example.part 3 0.5 0.25 0.25

# Run on CPU+GPUs
./main-dist example.graph example.part 0,1
```

License
===
This software is licensed under the GNU GPL v3.0.


Bibliography
===
[1] Heldens S., Varbanescu A. L., Prat-Pérez A. & Larriba-Pey J. L. Het-SCD: High-Quality Community Detection on Heterogeneous Platforms. Manuscript submitted for publication.

[2] Prat-Pérez, A., Dominguez-Sal, D., & Larriba-Pey, J. L. (2014, April). High quality, scalable and parallel community detection for large real graphs. In Proceedings of the 23rd international conference on World wide web (pp. 225-236). International World Wide Web Conferences Steering Committee.

