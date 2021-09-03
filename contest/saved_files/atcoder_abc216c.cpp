
/**
 * @author      : abhj
 * @created     : Thursday Sep 02, 2021 16:50:36 IST
 * @filename    : c.cpp
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
	cin >> n;
	while (n != 0) {
		if (n & 1 ^ 1) {
			s = 'B' + s;
			n >>= 1;
		}
		else {
			s = 'A' + s;
			n--;
		}
	}
	cout << s;
}

int32_t main() {
	ios_base::sync_with_stdio (false);
	cin.tie (0);
	solve();
	return 0;
}
