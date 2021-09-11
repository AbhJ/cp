

/**
 * @author      : abhj
 * @created     : Sunday Sep 05, 2021 20:08:18 IST
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

string s;

void solve() {
	cin >> s >> s;
	for (auto i : s) {
		if (i == 'D')
			cout << 'U';
		else if (i == 'U')
			cout << 'D';
		else
			cout << i;
	}
}

int32_t main() {
	ios_base::sync_with_stdio (false);
	cin.tie (0);
	int t;
	cin >> t;
	while (t--) {
		solve();
		cout << "\n";
	}
	return 0;
}
