#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>

#include "pperm.hh"

int main(int argc, char* argv[]) {
  int ch;
  opterr = 0;
  auto prog_name = argv[0];

  auto print_usage = [=]() {
    fprintf(stderr, "Usage: %s [-h] [-l length (default 10)] [-t test times (default 16)] algo1 algo2 ... algoN\n",
            prog_name);
    std::cerr << "Available algorithms: ";
    auto algorithms = PermAlgorithm::getNames();
    std::copy(algorithms.cbegin(), algorithms.cend(), std::ostream_iterator<const std::string>(std::cerr, " "));
    std::cerr << std::endl;
  };

  int n = 10;
  int n_test = 16;

  while ((ch = getopt(argc, argv, "hl:t:")) != -1) {
    switch (ch) {
      case 'l':
        n = atoi(optarg);
        break;
      case 't':
        n_test = atoi(optarg);
        break;
      case 'h':
        // falltrough
      case '?':
        print_usage();
        exit(ch != 'h');
        break;
    }
  }

  if (n <= 0) {
    fprintf(stderr, "Wrong permutation length: %d\n", n);
    exit(1);
  } else {
    fprintf(stderr, "Permutation length: %d\n", n);
  }

  if (n_test <= 1) {
    fprintf(stderr, "Wrong test times: %d\n", n_test);
  } else {
    fprintf(stderr, "Test repeating times: %d\n", n_test);
  }

  double n_compute = n;

  for (int i = 2; i <= n; ++i) {
    n_compute *= i;
  }

  if (optind == argc) {
    std::cerr << "No algorithm given" << std::endl;
    print_usage();
    exit(1);
  }

  for (int i = optind; i < argc; ++i) {
    std::string algo_name(argv[i]);
    auto algo = PermAlgorithm::get(algo_name);
    if (algo == nullptr) {
      fprintf(stderr, "No such algorithm: %s\n", algo_name.c_str());
      continue;
    }

    fprintf(stderr, "Setting up %s\n", algo_name.c_str());
    algo->setup(n);

    fprintf(stderr, "Warming up\n");
    algo->warmup();

    fprintf(stderr, "Testing\n");
    auto res = algo->benchmark(n_test);
    printf(
        "Algorithm %s, n = %d, mean = %.3lf ms, stddev = %.3lf ms, max = %.3lf ms, min = %.3lf ms, "
        "GEPs = %.3lf\n",
        algo_name.c_str(), n_test, res.mean * 1e3, res.stddev * 1e3, res.max * 1e3, res.min * 1e3,
        n_compute / res.mean * 1e-9);
  }
}
