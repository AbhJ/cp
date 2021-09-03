#include "bits/stdc++.h"
#define int          long long int
#define mp           make_pair
#define pb           emplace_back
#define F            first
#define S            second
using vi       =     std::vector<int>;
using vvi      =     std::vector<vi>;
using pii      =     std::pair<int, int>;
using vpii     =     std::vector<pii>;
using vvpii    =     std::vector<vpii>;
using namespace std;
const int inf  =     1e18 + 10;
const int N    =     5e6 + 10;
int n, a[N], ans; set<int> has[62], does_not_have[62]; char c[N];
int sz = 0, nxt[2][N], cn[N], koita[2][N]; bool exist[N], is_end[N];
void update(int val, int cnt) {
    int v = 0;
    for (int i = 61; i >= 0; i--) {
        int c = (val >> i) & 1LL;
        if (!exist[nxt[c][v]]) {
            nxt[c][v] = ++sz;
            exist[sz] = 1;
        }
        koita[c][v] += cnt;
        v = nxt[c][v];
    }
    is_end[v] = 1;
}
void query(int val) {
    int v = 0; ans = 0;
    for (int i = 61; i >= 0; i--) {
        int c = (val >> i) & 1LL;
        if (koita[c ^ 1LL][v] != 0) {
            v = nxt[c ^ 1LL][v];
            ans += (1LL << i);
        }
        else {
            v = nxt[c][v];
        }
    }
}
void f() {
    // trie solution
    for (int i = 1; i <= n; i++) {
        cin >> c[i] >> a[i];
        if (c[i] == '+')
            update(a[i], 1);
        if (c[i] == '-')
            update(a[i], - 1);
        if (c[i] == '?') {
            query(a[i]);
            cout << ans << "\n";
        }
    }
}
map<int, int>m;
void solve() {
    cin >> n;
    update(0, 1);
    if (1) {
        f();
        return;
    }
    for (int i = 1; i <= n; i++) {
        // n * n optimised brute
        cin >> c[i] >> a[i];
        m.clear();
        int ans = 0;
        for (int j = 61; j >= 0; j--) {
            if (c[i] == '+') {
                if ((a[i] >> j) & 1LL)
                    has[j].emplace(a[i]);
                else
                    does_not_have[j].emplace(a[i]);
            }
            if (c[i] == '-') {
                if ((a[i] >> j) & 1LL)
                    has[j].erase(a[i]);
                else
                    does_not_have[j].erase(a[i]);
            }
            if (c[i] == '?') {
                // TODO:    set the operations to find max xor
                if ((a[i] >> j) & 1LL)
                    for (auto val : does_not_have[j]) {
                        m[val] += (1LL << j);
                        ans = max(ans, m[val]);
                    }
                else {
                    for (auto val : has[j]) {
                        m[val] += (1LL << j);
                        ans = max(ans, m[val]);
                    }
                }
                if (j == 0) {
                    cout << ans << "\n";
                }
            }
        }
    }
}
int32_t main() {
    ios_base::sync_with_stdio(false); cin.tie(0);
    solve(); return 0;
    return 0;
}