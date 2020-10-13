#include <algorithm>
#include <pperm.hh>

class SJTCPUSimple : public PermAlgorithm {
 private:
  int *perm, *inv;
  bool *dir;

 protected:
  void setup_() override {
    perm = new int[n];
    inv = new int[n];
    dir = new bool[n];
    for (int i = 0; i < n; ++i) {
      perm[i] = i;
      inv[i] = i;
      dir[i] = false;
    }
  }

  void generate_() override {
    int s = 0;
    while (1) {
      int top = -1, nxt;
      for (int i(n - 1); i >= 0; i--) {
        nxt = inv[i] + (dir[i] ? 1 : -1);
        if (nxt >= 0 && nxt < n) {
          if (perm[nxt] < i) {
            top = i;
            break;
          }
        }
      }
      if (top == -1) {
        printf("???");
        break;
      }
      inv[perm[nxt]] = inv[top];
      perm[inv[top]] = perm[nxt];
      inv[top] = nxt;
      perm[nxt] = top;
      for (int i(top + 1); i < n; i++) dir[i] ^= 1;
      // for (int i(0); i<n; i++)
      //    printf("%d ",perm[i]+1);
      // puts("");
      s += 1;
    }
  }
};

REGISTER_PERM_ALGORITHM("sjt_cpu_simple", SJTCPUSimple)
