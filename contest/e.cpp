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
int n; string s;
pair<int, string> f(int x) {
	int taka = 0;
	string t;
	for (auto &i : s) {
		if (i != '?')
			t += i;
		else if (x ^ ((int)t.size() & 1))
			t += 'B';
		else t += 'R';
	}
	for (int i = 1; i < n; i++) {
		taka += t[i] == t[i - 1];
	}
	return mp(taka, t);
}
void solve() {
	cin >> n >> s;
	if (f(1).F < f(0).F)
		cout << f(1).S;
	else cout << f(0).S;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}