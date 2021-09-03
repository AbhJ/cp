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
int n, dp[N][5], tot[N], ma; string s[N];
void solve() {
	cin >> n;
	fill(s + 1, s + 1 + n, "");
	for (int i = 1; i <= n; i++) {
		fill(dp[i], dp[i] + 5, 0LL);
		cin >> s[i];
		for (auto &j : s[i])
			dp[i][j - 'a']++;
		tot[i] = (int)s[i].length();
	}
	ma = 0;
	for (int i = 0; i < 5; i++) {
		priority_queue<int> q;
		for (int j = 1; j <= n; j++) {
			q.emplace(dp[j][i] - (tot[j] - dp[j][i]));
		}
		int cnt = 0, z = 0;
		while (q.empty() == 0 and q.top() + cnt > 0)
			cnt += q.top(), q.pop(), z++;
		ma = max(ma, z);
	}
	cout << ma;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}