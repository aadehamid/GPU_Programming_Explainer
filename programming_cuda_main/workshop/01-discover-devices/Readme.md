# Exercise 1: Check for CUDA devices[^1]

## Goal:

The purpose of this exercise is to get familiar with the heterogeneous system we are programming. We will find out what kind of (GPE) devices are present on the system and what are their specifications.

## Instructions:

- move to subdierctory:
```console
$ cd 01-discover-devices/
```
- load CUDA module:
```console
$ module load CUDA
```
- build the application:
```console
$ make all
```

- Run the application on a compute node:
```console
$ srun --partition=gpu --ntasks=1 --gpus=1 prog
```

- You should obtain the following output:
```console
Device 0: "Tesla V100S-PCIE-32GB"
  GPU Clock Rate (MHz):                          1597
  Memory Clock Rate (MHz):                       1107
  Memory Bus Width (bits):                       4096
  Peak Memory Bandwidth (GB/s):                  1133.57
  CUDA Cores/MP:                                 64
  CUDA Cores:                                    5120
  Total amount of global memory:                 32 GB
  Total amount of shared memory per block:       48 kB
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
```

___

[^1]: 
 &copy; Patricio Bulić, Davor Sluga, University of Ljubljana, Faculty of computer and information science. 
 The materials are published under license [Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).

___