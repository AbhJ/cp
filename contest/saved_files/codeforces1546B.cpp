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
int n, m; string s[N];
void solve() {
	cin >> n >> m;
	map<int, int> cnt[400];
	for (int i = 0; i < (n << 1) - 1; i++)
		cin >> s[i];
	for (int i = 0; i < (n << 1) - 1; i++)
		for (int j = 0 ; j < m ; j++)
			cnt[s[i][j]][j]++;
	for (int i = 0; i < (n << 1) - 1; i++) {
		int F = 1;
		for (int j = 0; j < m; j++) F &= ((cnt[s[i][j]][j] & 1) == 1);
		if (F) {
			cout << s[i] + "\n";
			break;
		}
	}
	for (int i = 0; i < (n << 1) - 1; i++) s[i].clear();
	cout.flush();
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve();
	}
	return 0;
}