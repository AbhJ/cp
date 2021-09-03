
/**
 * @author      : abhj
 * @created     : Monday Aug 16, 2021 17:36:18 IST
 * @filename    : a.cpp
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

int g1, s1, b1, g2, s2, b2;

void solve() {
	cin >> g1 >> s1 >> b1 >> g2 >> s2 >> b2;
	cout << 1 + (g1 + s1 + b1 < g2 + s2 + b2);
}

int32_t main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	int t;
	cin >> t;
	while(t--) {
		solve();
		cout << "\n";
	}
	return 0;
}
