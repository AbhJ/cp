

/**
 * @author      : abhj
 * @created     : Wednesday Sep 08, 2021 20:23:44 IST
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
const int N    =     5e1 + 10;

int n;
string a[N];
string s;

void solve() {
	cin >> n >> s;
	set<int>done;
	for (int i = 0; i < n; i++) {
		a[i] = string (n, '.');
		a[i][i] = 'X';
		// if(s[i] == '1')
		// HE CANT LOSE ANY
		// if(s[i] == '2')
		// HE HAS TO WIN ATLEAST ONE
		for (int j = 0; j < i; j++) {
			a[i][j] = '.';
			if (s[i] == '1' and s[j] == '2')
				a[i][j] = '=';
			if (s[i] == '1' and s[j] == '1')
				a[i][j] = '=';
			if (s[i] == '2' and s[j] == '1')
				a[i][j] = '=';
			if (s[i] == '2' and s[j] == '2' and done.count (j))
				a[i][j] = '+', done.emplace (i);
			else if (s[i] == '2' and s[j] == '2')
				a[i][j] = '-', done.emplace (j);
			if (a[i][j] == '.') {
				cout << "NO";
				return;
			}
			if (a[i][j] == '+')
				a[j][i] = '-';
			if (a[i][j] == '-')
				a[j][i] = '+';
			if (a[i][j] == '=')
				a[j][i] = '=';
		}
	}
	for (int i = 0; i < n; i++) {
		if (
			(s[i] == '2' and count (a[i].begin(), a[i].end(), '+') == 0)
			or
			(s[i] == '1' and count (a[i].begin(), a[i].end(), '-') == 1)
		) {
			cout << "NO";
			return;
		}
	}
	cout << "YES\n";
	for (int i = 0; i < n; i++)
		cout << a[i] << "\n";
}

int32_t main() {
	ios_base::sync_with_stdio (false);
	cin.tie (0);
	int t;
	cin >> t;
	while (t--) {
		solve();
		cout << "\n";
	}
	return 0;
}
