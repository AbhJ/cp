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
const int N    =     2e6 + 10;
int n, m, t, x, y, ans, tmp, koi, q, cn[N]; vvi a;
void solve() {
	cin >> n >> m;
	for (int i = 1; i <= m; i++) {
		cin >> x >> y;
		cn[min(x - 1, y - 1)]++;
		// WHENEVER A NODE LOSES,
		// HE SURELY DIES
	}
	for (int i = 0; i < n; i++) {
		koi += !!cn[i];
	}
	cin >> q;
	for (int i = 1; i <= q; i++) {
		cin >> t;
		if (t != 3)
			cin >> x >> y;
		if (t == 1) {
			cn[min(x - 1, y - 1)]++;
			if (cn[min(x - 1, y - 1)] == 1)
				koi++;
		}
		if (t == 2) {
			cn[min(x - 1, y - 1)]--;
			if (cn[min(x - 1, y - 1)] == 0)
				koi--;
		}
		if (t == 3) {
			cout << n - koi << "\n";
		}
	}
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	solve(); return 0;
	return 0;
}