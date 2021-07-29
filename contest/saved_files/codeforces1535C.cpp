//Coded by Abhijay Mitra (AbJ)
#include "bits/stdc++.h"
#define double       long double
#define int          long long int
#define ibs          ios_base::sync_with_stdio(false)
#define cti          cin.tie(0)
#define mp           make_pair
#define pb           emplace_back
#define F            first
#define S            second
using vi       =     std::vector<int>;
using vvi      =     std::vector<vi>;
using pii      =     std::pair<int, int>;
using vpii     =     std::vector<pii>;
using vvpii    =     std::vector<vpii>;
#define rep(i, a, b) for (int i = a; i <= b; i++)
#define per(i, b, a) for (int i = b; i >= a; i--)
#define all(x)       x.begin(), x.end()
using namespace std;
const int inf  =     1e18 + 10;
const int N    =     2e6 + 10;
string s; int n, dp[2], ans;
void solve() {
	cin >> s;
	n = (int)s.length();
	dp[0] = dp[1] = ans = 0;
	// README
	// dp[j] means how many strings ending with j at i
	rep(i, 0, n - 1) {
		if (s[i] == '?') {
			swap(dp[0], dp[1]);
			dp[0] += 1;
			dp[1] += 1;
		}
		else {
			dp[(int)s[i] - '0'] = dp['1' - (int)s[i]] + 1;
			dp['1' - (int)s[i]] = 0;
		}
		ans += max(dp[0], dp[1]);
	}
	cout << ans;
}
int32_t main() {
	ibs; cti;
	// solve(); return 0;
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}