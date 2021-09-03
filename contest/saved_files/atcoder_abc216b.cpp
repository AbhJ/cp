
/**
 * @author      : abhj
 * @created     : Thursday Sep 02, 2021 16:31:39 IST
 * @filename    : b.cpp
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
string s[N], t[N];

void solve() {
	cin >> n;
	set<pair<string, string>>S;
	for (int i = 0; i < n; i++) {
		cin >> s[i] >> t[i];
		if (S.count (mp (s[i], t[i]))) {
			cout << "Yes";
			return;
		}
		S.emplace (mp (s[i], t[i]));
	}
	cout << "No";
}

int32_t main() {
	ios_base::sync_with_stdio (false);
	cin.tie (0);
	solve();
	return 0;
}
