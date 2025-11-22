# Exercise 2: vector sum[^1]

## Goal:

The purpose of this exercise is to run our first application, which uses a GPU to compute the sum of two vectors.

## Instructions:

- move to subdierctory:
```console
$ cd 02-vec-add-short/
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
