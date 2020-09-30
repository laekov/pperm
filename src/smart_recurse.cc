/*
 * Smart O(n!) recursion.
 * On i-th step, swap the i-th element with the rest elements in the array.
 * Should be faster than smart, but still slow because of too many recursion.
 */
#include <pperm.hh>
#include <algorithm>

class SmartRecurse: public PermAlgorithm {
private:
	int *a, s;

	void DFS(int cur) {
		if (cur == n) {
			s += 1;
			return;
		}
		for (int i = cur; i < n; ++i) {
			if (i == cur) {
				DFS(cur + 1);
				continue;
			}
			std::swap(a[i], a[cur]);
			DFS(cur + 1);
			std::swap(a[i], a[cur]);
		}
	}

protected:
	virtual void setup_() {
		a = new int[n];
		for (int i = 0; i < n; ++i) {
			a[i] = i;
		}
	}

	virtual void generate_() {
		s = 0;
		DFS(0);
	}
};

REGISTER_PERM_ALGORITHM("smart_recurse", SmartRecurse);
