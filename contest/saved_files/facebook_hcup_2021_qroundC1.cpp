

/**
 * @author      : abhj
 * @created     : Sunday Aug 29, 2021 11:33:30 IST
 * @filename    : d.cpp
 */

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
const int N    =     1e2 + 10;

vvi A;
int a[N], ma = - inf, n, x, y;
map <int, int> _map, pa;
map<int, vi>parent;
bool vis[100];
vi v;

void f (int ind, int val, int par) {
	int m = A[ind].size();
	if (ind == n + 1)
		return;
	for (auto j : A[ind]) {
		if (vis[j] == 0) {
			vis[j] = 1;
			f (j, val + a[j], ind);
		}
	}
	_map[ind] = val;
	pa[ind] = par;
	if (val > ma)
		ma = val;
	return;
}

void checkParent () {
	for (int i = 1; i <= n; i++) {
		int m = i;
		parent[i].pb (i);
		while (pa[m] > 1 ) {
			parent[i].pb (pa[m]);
			m = pa[m];
		}
		sort (parent[i].begin(), parent[i].end());
	}
}

void solve() {
	int ans = - inf;
	cin >> n;
	for (int i = 1; i <= n; i++)
		cin >> a[i];
	A = vvi (N);
	memset (vis, 0, sizeof (vis));
	pa.clear();
	_map.clear();
	v.clear();
	parent.clear();
	for (int i = 1; i <= n - 1; i++) {
		cin >> x >> y;
		A[x].pb (y);
		A[y].pb (x);
	}
	ma = - inf;
	vis[1] = 1;
	f (1, a[1], -1);
	checkParent();
	for (int i = 2; i <= n; i++) {
		for (int j = i + 1; j <= n; j++) {
			for (auto a : parent[i]) {
				if (binary_search (parent[j].begin(), parent[j].end(), a))
					goto F;
			}
			ans = max (ans, _map[i] + _map[j] - a[1]);
F:
			;
		}
	}
	cout << (ans == - inf ? ma : ans);
}

int32_t main() {
	ios_base::sync_with_stdio (false);
	cin.tie (0);
	int t, x = 0;
	cin >> t;
	while (t--) {
		cout << "Case #" << ++x << ": ";
		solve();
		cout << "\n";
	}
	return 0;
}
