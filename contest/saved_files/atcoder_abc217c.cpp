
/**
 * @author      : abhj
 * @created     : Sunday Sep 05, 2021 18:09:04 IST
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

int n, p[N], q[N];

void solve() {
	cin >> n;
	for (int i = 0; i < n; i++) {
		cin >> p[i];
		q[p[i] - 1] = i + 1;
	}
	for (int i = 0; i < n; i++)
		cout << q[i] << " ";
}

int32_t main() {
	ios_base::sync_with_stdio (false);
	cin.tie (0);
	solve();
	return 0;
}
