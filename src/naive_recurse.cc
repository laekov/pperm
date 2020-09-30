/* 
 * Naive O(n! * n) recursion.
 * Should be slow.
*/
#include <pperm.hh>

class NaiveRecurse: public PermAlgorithm {
private:
	int *a, s;

	void DFS(int cur, int occupied) {
		if (cur == n) {
			s += 1;
			return;
		}
		for (int i = 0; i < n; ++i) {
			if (occupied & (1 << i)) {
				continue;
			}
			a[cur] = i;
			DFS(cur + 1, occupied | (1 << i));
		}
	}

protected:
	virtual void setup_() {
		a = new int[n];
	}

	virtual void generate_() {
		s = 0;
		DFS(0, 0);
	}
};

REGISTER_PERM_ALGORITHM("naive_recurse", NaiveRecurse);
