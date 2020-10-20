# Parallel Permutation Generator

Project I of Zuhe Math Course by bakser, harry, laekov

## Introduction

We study four established permutation generation methods, including the naive recursive method, Heap's algorithm, lexicographic generation, and Steinhaus–Johnson–Trotter algorithm. 

We purpose a novel way to distribute the enumeration process evenly to multiple computation units, and process them in parallel.

We implement and profile them on a CPU cluster and a GPU. We observe that using a single server, up to 149x speed up is gained against the single thread implementation. 

## Building and contributing instructions

### Building and running

```
mkdir build
cd build
cmake ..
make
(maybe numactl or other env here) ./pperm %s [-h] [-l length (default 10)] [-t test times (default 16)] [-d CPU distribution factor (default 10)] algo1 algo2 ... algoN
```

### Adding a new algorithm

Duplicate `src/lex/std.cc` for a new algorithm and change the code. 

Remember to change the last line of the source file to register the new algorithm properly.

