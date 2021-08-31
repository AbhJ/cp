/**
 * @author      : abhj
 * @created     : Friday Aug 27, 2021 22:49:27 IST
 * @filename    : a.cpp
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

int ok (char x) {
	return x == 'A' or x == 'E' or x == 'I' or x == 'O' or x == 'U';
}

int n, con, vow, ans;
string s;

void solve() {
	cin >> s;
	n = s.length();
	vow =     count (s.begin(), s.end(), 'A')
			  + count (s.begin(), s.end(), 'E')
			  + count (s.begin(), s.end(), 'I')
			  + count (s.begin(), s.end(), 'O')
			  + count (s.begin(), s.end(), 'U');
	vector<pair<int, char>>v{
		mp (count (s.begin(), s.end(), 'A'), 'A'),
		mp (count (s.begin(), s.end(), 'E'), 'E'),
		mp (count (s.begin(), s.end(), 'I'), 'I'),
		mp (count (s.begin(), s.end(), 'O'), 'O'),
		mp (count (s.begin(), s.end(), 'U'), 'U')
	};
	map<char, int>m;
	int count = 0;
	ans = inf;
	for (int i = 0; i < n; i++) {
		if (ok (s[i]))
			count++;
		m[s[i]]++;
	}
	for (char i = 'A'; i <= 'Z'; i++) {
		if (ok (i))
			ans = min (ans, 2 * (count - m[i]) + n - count);
		else
			ans = min (ans, count + 2 * (n - count - m[i]));
	}
	cout << ans;
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
