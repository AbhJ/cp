/**
 * @author      : abhj
 * @created     : Sunday Aug 29, 2021 12:05:25 IST
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

string s, t;
int n, m, k, ans = inf, tmp_ans, koi, cnt;

void solve() {
	cin >> s >> k;
	m = s.length();
	vvi dp (26, vi (26, inf));
	// FLOYD WARSHALL
	for (int i = 0; i < k; i++) {
		cin >> t;
		dp[t[0] - 'A'][t[1] - 'A'] = 1;
	}
	for (k = 0; k < 26; k++)
		for (int i = 0; i < 26; i++)
			for (int j = 0; j < 26; j++)
				if (dp[i][j] > (dp[i][k] + dp[k][j]) and (dp[k][j] != inf and dp[i][k] != inf))
					dp[i][j] = dp[i][k] + dp[k][j];
	for (char j = 'A'; j <= 'Z'; j++) {
		tmp_ans = 0;
		for (int i = 0; i < m; i++)
			if (s[i] != j) {
				int step = dp[s[i] - 'A'][j - 'A'];
				if (step == inf)
					goto F
					;
				else
					tmp_ans += step;
			}
		ans = min (ans, tmp_ans);
F:
		;
	}
	cout << (ans == inf ? - 1 : ans);
	ans = inf;
}

int32_t main() {
	ios_base::sync_with_stdio (false);
	cin.tie (0);
	int t, x = 0;
	cin >> t;
	while (t--) {
		cout << "Case #" << ++x << ": ";
		solve();
		cout << "\n";
	}
	return 0;
}
