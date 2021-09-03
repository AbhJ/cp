

/**
 * @author      : abhj
 * @created     : Sunday Aug 29, 2021 14:24:25 IST
 * @filename    : c.cpp
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
const int N    =     5e1 + 10;

int n;
char a[N][N];
set<int> r, c;
map <int, set<set<pii>>>s;

void solve() {
	cin >> n;
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) {
			cin >> a[i][j];
			if (a[i][j] != 'O')
				r.emplace (i), c.emplace (j);
		}
	for (auto i : r) {
		int cnt = 0;
		set<pii> v;
		for (int j = 0; j < n; j++) {
			if (a[i][j] == 'O')
				goto F;
			if (a[i][j] == '.')
				cnt++, v.emplace (mp (i, j));
		}
		s[cnt].emplace (v);
F:
		;
	}
	for (auto j : c) {
		int cnt = 0;
		set<pii> v;
		for (int i = 0; i < n; i++) {
			if (a[i][j] == 'O')
				goto FF;
			if (a[i][j] == '.')
				cnt++, v.emplace (mp (i, j));
		}
		s[cnt].emplace (v);
FF:
		;
	}
	if (s.empty() == 1)
		cout << "Impossible";
	else
		cout << s.begin() -> F << " " << s.begin() -> S.size();
	s.clear();
	r.clear();
	c.clear();
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
