# Exercise 3: sum of two vectors of arbitrary length[^1]

## Goal:

The purpose of this exercise is to run an application, which uses a GPU to compute the sum of two vectors of arbitrary lengths.

## Instructions:

- move to subdierctory:
```console
$ cd 03-vec-add-arb/
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

