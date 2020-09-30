/* 
 * Naive O(n! * n) recursion with manually implemented stack.
 * Expected to save great amount of function call overhead.
 * But it cannot really generate the permutation.
*/
#include <pperm.hh>

class SimStackRecurse: public PermAlgorithm {
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

REGISTER_PERM_ALGORITHM("simstack_recurse", SimStackRecurse);
