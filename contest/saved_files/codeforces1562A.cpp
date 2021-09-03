
/**
 * @author      : abhj
 * @created     : Thursday Aug 26, 2021 20:06:57 IST
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

int a, b;

void solve() {
	cin >> a >> b;
	cout << b % min (max (b / 2 + 1, a), b);
}

int32_t main() {
	ios_base::sync_with_stdio (false);
	int t;
	cin >> t;
	cin.tie (0);
	while (t--) {
		solve();
		cout << "\n";
	}
	return 0;
}
