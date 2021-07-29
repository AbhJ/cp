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
int k, n, cn, a[3000], dp[63], taka;
void solve() {
	cin >> k;
	cn = 0;
	if (k & 1) {
		cout << - 1;
		return;
	}
	memset(a, 0, sizeof a);
	for (int i = 1; i <= inf; i++) {
		int bit = prev(upper_bound(dp + 1, dp + 63, k)) - dp;
		k -= dp[bit];
		cn += bit;
		a[cn] = 1;
		if (k <= 0) {
			assert(k == 0);
			cout << cn << "\n1 ";
			for (int j = 1; j < cn; j++) {
				cout << a[j] << " ";
			}
			return;
		}
	}

}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	for (int i = 1; i <= 62; i++) {
		dp[i] = (dp[i - 1] + 1LL) << 1LL;
	}
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}