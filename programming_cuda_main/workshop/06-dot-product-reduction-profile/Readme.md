# Exercise 6: Measure kernel time of the dot product[^1]

## Goal:

The purpose of this exercise is to get familiar with measuring the execution time of CUDA kernels.

## Instructions:

- move to subdierctory:
```console
$ cd 06-dot-product-reduction-profile/
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

___

[^1]: 
 &copy; Patricio Bulić, Davor Sluga, University of Ljubljana, Faculty of computer and information science. 
 The materials are published under license [Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).

___
