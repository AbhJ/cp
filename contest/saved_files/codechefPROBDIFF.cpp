

/**
 * @author      : abhj
 * @created     : Monday Aug 16, 2021 17:49:48 IST
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

vi a;

void solve() {
	for(int i = 0 ; i < 4; i++) {
		int x;
		cin >> x;
		a.pb(x);
	}
	int cnt = 0;
	sort(a.begin(), a.end());
	if(a.front() == a.back()) {
		cout << 0;
	}
	else if(a.front() == a[2] or a[1] == a.back()) {
		cout << 1;
	}
	else
		cout << 2;
	a.clear();
}

int32_t main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	int t;
	cin >> t;
	while (t--) {
		solve();
		cout << "\n";
	}
	return 0;
}
