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
(maybe numactl or other env here) ./pperm %s [-h] [-l length (default 10)] [-t test times (default 16)] [-d CPU distribution factor (default 10)] algo1 algo2 ... algoN
```

### Adding a new algorithm

Duplicate `src/lex/std.cc` for a new algorithm and change the code. 

Remember to change the last line of the source file to register the new 
algorithm properly.
