/**
 * @author      : abhj
 * @created     : Friday Sep 03, 2021 23:08:16 IST
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

string s, t;

string f (bool a) {
	if (a)
		return "Yes";
	else
		return "No";
}

void solve() {
	cin >> s >> t;
	cout << f (s < t);
}

int32_t main() {
	ios_base::sync_with_stdio (false);
	cin.tie (0);
	solve();
	return 0;
}
