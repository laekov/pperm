#include "pperm.hh"


int main(int argc, char* args[]) {
	// TODO: Use getopt here
	std::string algo_name = argc >= 2 ? std::string(args[1]) : "Std";
	int n = argc >= 3 ? atoi(args[2]) : 10;
	
	auto algo = PermAlgorithmRegistry::get(algo_name);
	if (algo == nullptr) {
		fprintf(stderr, "No such algorithm: %s\n", algo_name.c_str());
		return 1;
	}

	fprintf(stderr, "Setting up\n");
	algo->setup(n);

	fprintf(stderr, "Warming up\n");
	algo->warmup();

	fprintf(stderr, "Testing\n");
	auto res = algo->benchmark();
	printf("Algorithm %s, n = %d, mean = %.3lf ms, max = %.3lf\n", 
			algo_name.c_str(), n, res.first * 1e3, res.second * 1e3);
}
