Parallel Permutation Generator
===
Project I of Zuhe Math Course by bakser, harry, laekov

## Introduction

No zhidao

## Building and contributing instructions

### Building and running

```
mkdir build
cd build
cmake ..
make
(maybe numactl or other env here) ./pperm <Algorithm name> <n>
```

### Adding a new algorithm

Duplicate `src/std.cc` for a new algorithm and change the code. 

Remember to change the last line of the source file to register the new 
algorithm properly.
