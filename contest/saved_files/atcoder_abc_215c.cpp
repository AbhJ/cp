
/**
 * @author      : abhj
 * @created     : Saturday Aug 21, 2021 23:51:49 IST
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
const int N    =     2e6 + 10;

string s; int k, n;

void solve() {
	cin >> s >> k;
	n = s.length();
	assert(n <= 8 and n >= 1);
	sort(s.begin(), s.end());
	while(--k){
		assert(next_permutation(s.begin(), s.end()) == 1);
	}
	cout << s;
}

int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	solve();
	return 0;
}
