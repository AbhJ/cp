
/**
 * @author      : abhj
 * @created     : Saturday Aug 21, 2021 23:17:04 IST
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
const int N    =     2e6 + 10;

int cnt, n, m, a[N], vis[N];

vi fact(int n) {
	vi v;
	for (int i = 1; i * i <= n; ++i) {
		if (n % i == 0) {
			v.pb(i);
			if (i * i != n) {
				v.pb(n / i);
			}
		}
	}
	return v;
}

void solve() {
	cin >> n >> m;

	for(int i = 0; i < n; i++) {
		cin >> a[i];
		vi fac = fact(a[i]);
		for(auto &j: fac) {
			vis[j] = 1;
		}
	}

	vi v{1};

	for(int i = 2; i <= m; i++) {
		if(vis[i] == 1)
			for(int j = i * 2; j <= m; j +=i) {
				vis[j] = 1;
			}
		else {
			v.pb(i);
		}
	}

	cout << v.size() << "\n";
	for(auto &i: v)
		cout << i << "\n";
}

int32_t main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	solve();
	return 0;
}
