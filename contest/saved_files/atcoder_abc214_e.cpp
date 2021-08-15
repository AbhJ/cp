
/**
 * @author      : abhj
 * @created     : Saturday Aug 14, 2021 23:21:43 IST
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

pii a[N];
priority_queue<int, vi, greater<int>> q;
int n;

void solve() {
	cin >> n;
	for (int i = 0; i < n; i++) {
		cin >> a[i].F >> a[i].S;
	}
	a[n] = make_pair(inf, inf);
	sort(a, a + n);
	int done = 0;
	for (int i = 0; i < n + 1; i++) {
		while (done < a[i].F and q.empty() == 0) {
			int en = q.top();
			q.pop();
			if (en < done) {
				cout << "No";
				return;
			}
			done++;
		}
		q.emplace(a[i].S);
		assert(a[i].F >= done);
		done = a[i].F;
	}
	cout << "Yes";
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
