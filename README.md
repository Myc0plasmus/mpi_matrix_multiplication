# MPI matrix multiplication

This repository contains few variants of cannon alogrithm for matrix multiplication implemented in pure C using MPI. 

## Prerequsites 

Nix-shell with necessary environment can be entered via:
```
nix-shell
```

## Predefined constants
Some important constants are defined at the beginning of each file
P - number of processes
N - dimensions of multiplied matrix (provided as single integer since it is assume that it is a square matrix)
PP - square root of number of processes
K - number of computers used for computation

## Description
Algorithm variants take as an input file named `liczby.txt` which contain two matrices. Then after preprocessing - matrix multiplication is performed via cannon algorithm, then sequentially. Processing and calculation time is measured for both. 

Then some statistics about the run are printed - comparison between sequential and parallel execution. Some variants save the results in the csv file in a form of a table. 

## Variant description

* `mpi_matrix_multiplication.c` - initial template variant, during cannon algorithm waits only for receive, while gathering result blocks from other processes uses `MPI_Type_create_subarray` - which is problematic in multi-computer calculations. Doesn't save result to csv.
* `mpi_kom_matrix_multiplication.c` - during cannon algorithm waits only for receive, while gathering result blocks from other processes sends arrays elemnt by element. Doesn't save result to csv.
* `mpi_row_matrix_multiplication.c` - during cannon algorithm waits only for receive, while gathering result blocks from other processes sends arrays row by row. Does save result to csv.
* `mpi_alt_matrix_multiplication.c` - during cannon algorithm waits only for receive, while gathering result blocks from other processes sends whole arrays using pointer arithmetic. Does save result to csv.
* `mpi_swt_matrix_multiplication.c` - during cannon algorithm waits only for both receive and send, while gathering result blocks from other processes sends whole arrays using pointer arithmetic. Does save result to csv.
* `mpi_src_matrix_multiplication.c` - during cannon algorithm exchanges blocks with `MPI_Sendrecv`, while gathering result blocks from other processes sends whole arrays using pointer arithmetic. Does save result to csv.

## Utilities

* `dataGenerator.c` - generates matrices in suitable format for all variants
* `mpi_seq_matrix_multiplication.c` - compares IKJ and IJK orders of sequential matrix multiplication




