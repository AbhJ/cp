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
const int N    =     2e3 + 10;
int n, k, a[N], dp[N][N];
void solve() {
	cin >> n >> k;
	for (int i = 1; i <= n; i++) {
		cin >> a[i];
	}
	memset(dp, 0, sizeof dp);
	for (int i = 1; i <= n; i++) {
		for (int j = 0; j <= i; j++) {
			// j IS NUMBER OF ELEMENTS TO DELETE
			// dp[i][j] IS THE NUMBER OF ELEMENTS AT
			// CORRECT POSITION

			// IF WE DELETE
			// if (j != 0)
			// 	dp[i][j] = dp[i - 1][j - 1];

			// IF WE DON'T DELETE THE ith ELEMENT
			// dp[i][j] = dp[i - 1][j] + (a[i] == i - j);

			dp[i][j] = max(j == 0 ? 0 : dp[i - 1][j - 1], dp[i - 1][j] + (a[i] == i - j));
		}
	}
	for (int j = 0; j <= n; j++)
		if (dp[n][j] >= k) {
			cout << j;
			return;
		}
	cout << - 1;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}