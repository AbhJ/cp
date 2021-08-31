
/**
 * @author      : abhj
 * @created     : Tuesday Aug 24, 2021 15:12:03 IST
 * @filename    : f.cpp
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

vpii byX, byY;
int x, y, l, r, m, n;

static bool cmpX(pii a, pii b) {
	return a.F < b.F;
}

static bool cmpY(pii a, pii b) {
	return a.S < b.S;
}

bool f(int val) {
	queue<pii>q;
	int mi = min(byY.back().S - byY.front().S, byX.back().F - byX.front().F), ma = 0;
	for(auto [i, j]: byX) {
		int reqX = i - val, reqYBig = j + val, reqYSma = j - val;
		//q.top().F WILL ALWAYS BE SMALLER THAN i
		while(q.empty() == 0 and q.front().F <= reqX) {
			mi = min(mi, q.front().S);
			ma = max(ma, q.front().S);
			q.pop();
		}
		q.emplace(i, j);
		if(ma >= reqYBig or mi <= reqYSma)
			return 1;
	}
	return 0;
}

void solve() {
	cin >> n;
	for(int i = 1; i <= n; i++) {
		cin >> x >> y;
		byX.pb(x, y);
		byY.pb(x, y);
	}
	sort(byX.begin(), byX.end(), cmpX);
	sort(byY.begin(), byY.end(), cmpY);
	l = 0, r = min(byY.back().S - byY.front().S, byX.back().F - byX.front().F);
	while(r > l) {
		m = l + r + 1 >> 1;
		if(f(m))
			l = m;
		else
			r = m - 1;
	}
	cout << l;
}

int32_t main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	solve();
	return 0;
}
