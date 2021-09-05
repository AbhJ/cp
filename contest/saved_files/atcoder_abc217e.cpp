
/**
 * @author      : abhj
 * @created     : Sunday Sep 05, 2021 18:58:57 IST
 * @filename    : e.cpp
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

deque<int>v;
multiset<int>s;

int n, x, q;

void solve() {
	cin >> q;
	while (q--) {
		cin >> n;
		switch (n) {
		case 1:
			cin >> x;
			v.emplace_back (x);
			break;
		case 2:
			if (s.empty()) {
				cout << v.front() << "\n";
				v.pop_front();
			}
			else {
				cout << *s.begin() << "\n";
				s.erase (s.begin());
			}
			break;
		case 3:
			while (v.empty() == 0) {
				s.emplace (v.back());
				v.pop_back();
			}
			break;
		default:
			exit (1);
		}
	}
}

int32_t main() {
	ios_base::sync_with_stdio (false);
	cin.tie (0);
	solve();
	return 0;
}
