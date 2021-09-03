/**
 * @author      : abhj
 * @created     : Thursday Sep 02, 2021 18:48:24 IST
 * @filename    : d.cpp
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

vector<queue<int>> cyl;
int n, m, x;
vvi a;
queue<int>q;

void solve() {
	cin >> n >> m;
	cyl.resize (m);
	a.resize (n);
	int I = 0;
	for (auto &i : cyl) {
		cin >> x;
		while (x--) {
			int col;
			cin >> col;
			i.emplace (col - 1);
		}
		a[i.front()].pb (I);
		if (a[i.front()].size() == 2)
			q.emplace (i.front());
		I++;
	}
	int cnt = 0;
	while (q.empty() == 0) {
		int col = q.front();
		q.pop();
		cnt++;
		if (a[col].empty() == 0)
			for (auto &i : a[col]) {
				cyl[i].pop();
				if (cyl[i].empty() == 0) {
					a[cyl[i].front()].pb (i);
					if (a[cyl[i].front()].size() == 2)
						q.emplace (cyl[i].front());
				}
			}
	}
	for (auto &i : cyl) {
		if (i.empty() == 0) {
			// cout << i.front();
			cout << "No";
			return;
		}
	}
	// a[i] IS THE ADJACENCY LIST OF INDICES OF CYLINDER THAT HAVE i AS FIRST ELEMENT
	cout << "Yes";
}
int32_t main() {
	ios_base::sync_with_stdio (false);
	cin.tie (0);
	solve();
	return 0;
}
