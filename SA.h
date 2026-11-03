#include <bits/stdc++.h>
#include "Config.h"
using namespace std;

using ll = long long;


struct SAConfig {
    double T0 = 1000;
    double Tmin = 1e-3;
    double cooling = 0.995;
    int inner = 100;
};

class SA_solution {
  public:
    mt19937 rng{(chrono::steady_clock::now().time_since_epoch().count())};

    double rand01() { return uniform_real_distribution<double>(0, 1)(rng); }

    // 需要自己实现：计算方案的目标值
    double evaluate(const vector<int> &s) {
        // TODO
        return 0;
    }

    // 需要自己实现：从当前方案生成邻居
    vector<int> neighbor(vector<int> s) {
        // TODO：例如随机交换两个位置
        return s;
    }

    vector<int> SA(vector<int> cur, SAConfig x) {
        vector<int> best = cur;
        double curVal = evaluate(cur);
        double bestVal = curVal;
    
        while (x.T0 > x.Tmin) {
            for (int i = 0; i < 100; i++) {
                vector<int> nxt = neighbor(cur);
                double nxtVal = evaluate(nxt);
                double delta = nxtVal - curVal;

                // 更优必收；更差按概率接受
                if (delta < 0 || rand01() < exp(-delta / x.T0)) {
                    cur = nxt;
                    curVal = nxtVal;
                }

                if (curVal < bestVal) {
                    best = cur;
                    bestVal = curVal;
                }
            }

            x.T0 *= x.cooling; // 降温
        }
        return best;
    }
};