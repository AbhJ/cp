
/**
 * @author      : abhj
 * @created     : Sunday Sep 05, 2021 18:17:50 IST
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

int l, q, c[N], x[N];

set<int> s{0};

void solve() {
	cin >> l >> q;
	s.emplace (l);
	while (q--) {
		cin >> c[q] >> x[q];
		if (c[q] == 1)
			s.emplace (x[q]);
		else
			cout << *s.upper_bound (x[q]) - *prev (s.upper_bound (x[q])) << "\n";
	}
}

int32_t main() {
	ios_base::sync_with_stdio (false);
	cin.tie (0);
	solve();
	return 0;
}
