
/**
 * @author      : abhj
 * @created     : Wednesday Sep 08, 2021 20:08:50 IST
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

int n;
string s;

void solve() {
	cin >> n >> s;
	for (int i = 0; i < n; i++) {
		int c = 0;
		for (int j = i; j < n; j++) {
			c += s[j] == 'a';
			c -= s[j] == 'b';
			if (c == 0) {
				cout << i + 1 << " " << j + 1;
				return;
			}
		}
	}
	cout << "-1 -1";
}

int32_t main() {
	ios_base::sync_with_stdio (false);
	cin.tie (0);
	int t;
	cin >> t;
	while (t--) {
		solve();
		cout << '\n';
	}
	return 0;
}
