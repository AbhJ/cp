

/**
 * @author      : abhj
 * @created     : Sunday Aug 15, 2021 20:30:37 IST
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
	queue<pair<int, char>>q;
	for (int i = 0; i < n; i++) {
		if (s[i] == 'B') {
			q.emplace(i - 1, 'R');
			q.emplace(i + 1, 'R');
		}
		if (s[i] == 'R') {
			q.emplace(i - 1, 'B');
			q.emplace(i + 1, 'B');
		}
	}
	if (s == "?") {
		cout << "B";
		return;
	}

	int cn = 0;

	if (count(s.begin(), s.end(), '?') != (int)s.length())


		while (++cn < 200 and count(s.begin(), s.end(), '?') > 1) {
			for (int i = 0; i < n - 1; i++) {
				if (s[i] == 'B') {
					if (i != 0 and s[i - 1] == '?')
						s[i - 1] = 'R';
					if (s[i + 1] == '?')
						s[i + 1] = 'R';
				}
				if (s[i] == 'R') {
					if (i != 0 and s[i - 1] == '?')
						s[i - 1] = 'B';
					if (s[i + 1] == '?')
						s[i + 1] = 'B';
				}
			}
		}

	if (f(1).F < f(0).F)
		cout << f(1).S;
	else cout << f(0).S;

}

int32_t main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	int t;
	cin >> t;
	while (t--) {
		solve();
		cout << "\n";
	}
	return 0;
}
