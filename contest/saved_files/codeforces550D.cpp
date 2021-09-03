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
int k; vi a[N];
void solve() {
	cin >> k;
	vpii v;
	if (k == 1) {
		cout << "YES\n2 1\n1 2\n";
		return;
	}
	v.pb(k + 2, k + 3);
	// README
	// every member in connected of k + 2 nodes has k + 1 nodes to connect to
	// purposefully we leave one of them (node i + 1 for the node i) so that
	// degree remains k
	vi cn(2 * k + 5);
	for (int i = 1; i <= k + 2; i++) {
		for (int j = i + 1; j <= k + 2; j++) {
			if (i != 1 and i & 1 and j == i + 1)
				// skipping consecutive one as in README
				continue;
			// 1st one has k + 1 nodes to choose out of which it leaves last node
			// if it leaves 2 nd node, it would be disconnected from n for k == 2
			if (i <= 2 and j == k + 2)
				continue;
			v.pb(i, j);
			cn[i]++;
			cn[j]++;
			v.pb(2 * k + 5 - i, 2 * k + 5 - j);
			cn[2 * k + 5 - i]++;
			cn[2 * k + 5 - j]++;
		}
	}
	cn[k + 2]++;
	cn[k + 3]++;
	if (count(cn.begin(), cn.end(), k) != 2 * k + 4) {
		cout << "NO";
		return;
	}
	cout << "YES\n" << 2 * k + 4 << " " << (int)v.size() << "\n";
	for (int i = 0; i < (int)v.size(); i++) {
		cout << v[i].F << " " << v[i].S << "\n";
	}
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	solve(); return 0;
	return 0;
}