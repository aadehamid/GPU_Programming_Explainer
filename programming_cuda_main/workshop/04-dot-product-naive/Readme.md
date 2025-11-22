# Exercise 4: naive dot product[^1]

## Goal:

The purpose of this exercise is to implement a simple dot product of two vectors on the GPU.

## Instructions:

- move to subdierctory:
```console
$ cd 04-dot-product-naive/
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
