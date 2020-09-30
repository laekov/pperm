#include <algorithm>
#include <pperm.hh>

class Std: public PermAlgorithm {
private:
	int *a;

protected:
	virtual void setup_() {
		a = new int[n];
	}

	virtual void generate_() {
		for (int i = 0; i < n; ++i) {
			a[i] = i;
		}
		int s = 0, is_end = 0;
		while (!is_end) {
			s += 1;
			is_end = 1;
			for (int i = 0; i + 1 < n; ++i) {
				if (a[i] < a[i + 1]) {
					is_end = 0;
					break;
				}
			}
			std::next_permutation(a, a + n);
		}
	}
};

REGISTER_PERM_ALGORITHM("Std", Std);
