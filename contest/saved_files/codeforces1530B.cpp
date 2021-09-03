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
int a[21][21], n, m;
void solve() {
	cin >> n >> m;
	memset(a, 0LL, sizeof a);
	for (int i = 1; i < m; i++)a[n - 1][m - i - 1] = (i % 2 == 0) ? 1 : 0;
	for (int i = 1; i < n - 1; i++)a[i][0] = i % 2 == 0 ? 1 : 0;
	for (int i = 1; i < n; i++)a[i][m - 1] = i % 2 == 0 ? 1 : 0;
	for (int i = 0; i < m; i++)a[0][i] = (i % 2 == 0 ? 1 : 0);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++)
			if (i == n - 2 and j == 0) cout << 0;
			else cout << a[i][j];
		cout << "\n";
	}
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve();
	}
	return 0;
}