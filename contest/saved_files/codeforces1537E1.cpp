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
int n, k; string ans = string(1, (char)('z' + 1)), s;
void f(string s) {
	string st;
	for (int i = 0; i < k; i++)
		st += s[i % (int)s.size()];
	ans = min(st, ans);
}
void solve() {
	cin >> n >> k >> s;
	for (int i = 1; i <= n; i++) {
		f(s.substr(0, i));
	}
	cout << ans;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	solve();
	return 0;
}