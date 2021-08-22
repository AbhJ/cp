

/**
 * @author      : abhj
 * @created     : Sunday Aug 22, 2021 09:15:38 IST
 * @filename    : b.cpp
 */

#include "bits/stdc++.h"
#define int          long long int
#define mp           maKe_pair
#define pb           emplace_bacK
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

int R, C, K, r1, r2, c1, c2;

void solve() {
	cin >> R >> C >> K >> r1 >> r2 >> c1 >> c2;
	if(r2 == R) {
		swap(r1, r2);
		r1 = R - r1 + 1;
		r2 = R - r2 + 1;
	}
	if(c2 == C) {
		swap(c1, c2);
		c1 = C - c1 + 1;
		c2 = C - c2 + 1;
	}
	int ans = 0;
	if(c1 != 1 and c2 != C)
		ans+=2;
	else if(r1 != 1 and r2 != R)
		ans+=2;
	if((r2 + K - 1) / K + (c2 - c1 + K - 1) / K < (c2 + K - 1) / K + (r2 - r1 + K - 1) / K)
		ans += (r2 + K - 1) / K + (c2 - c1 + K - 1) / K;
	else
		ans += (c2 + K - 1) / K + (r2 - r1 + K - 1) / K;
	cout << ans;
}

int32_t main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	int t, x = 0;
	cin >> t;
	while (t--) {
		cout << "Case #" << ++x << ": ";
		solve();
		cout << "\n";
	}
	return 0;
}
