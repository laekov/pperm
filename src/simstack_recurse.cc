/* 
 * Naive O(n! * n) recursion with manually implemented stack.
 * Expected to save great amount of function call overhead.
 * But it cannot really generate the permutation.
*/
#include <pperm.hh>
#include <algorithm>

class NaiveFakeRecurse: public PermAlgorithm {
private:
	int *stack, top, s;

protected:
	virtual void setup_() {
		stack = new int[n * n];
	}

	virtual void generate_() {
		s = 0;
		stack[top = 0] = 0;
		while (top >= 0) {
			int occ = stack[top--];
			if (occ == ((1 << n) - 1)) {
				s += 1;
				continue;
			}
			for (int i = 0; i < n; ++i) {
				if (occ & (1 << i)) {
					continue;
				}
				stack[++top] = occ | (1 << i);
			}
		}
	}
};

REGISTER_PERM_ALGORITHM("naive_fakerecurse", NaiveFakeRecurse);

/*
 * Use manually implemented stack to run the O(n!) algorithm
 * Expected to be fast
 */

class SmartFakeRecurse: public PermAlgorithm {
private:
	int *a, *stack, top, s;

protected:
	virtual void setup_() {
		a = new int[n];
		for (int i = 0; i < n; ++i) {
			a[i] = i;
		}
		stack = new int[n];
	}

	virtual void generate_() {
		s = 0;
		stack[top = 0] = 0;
		while (top >= 0) {
			if (top == n) {
				s += 1;
				/* TODO: do some callback
				for (int i = 0; i < n; ++i) {
					printf("%d ", a[i]);
				}
				putchar(10);
				*/
			}
			if (top == n || stack[top] == n) {
				--top;
				if (top >= 0) {
					if (++stack[top] < n) {
						if ((n - top) & 1) {
							std::swap(a[top], a[n - 1]);
						} else {
							std::swap(a[top], a[stack[top]]);
						}
					}
				}
				continue;
			}
			if (top + 1 < n) {
				stack[top + 1] = top + 1;
			}
			++top;
		}
	}
};

REGISTER_PERM_ALGORITHM("smart_fakerecurse", SmartFakeRecurse);
