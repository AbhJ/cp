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
const int N    =     2e5 + 10;
int n, k, a[N], col[N]; vi ind[N], v;
void solve() {
	cin >> n >> k;
	fill(ind + 1, ind + 1 + n, vi());
	fill(col + 1, col + 1 + n, 0LL);
	v.clear();
	for (int i = 1; i <= n; i++) {
		cin >> a[i];
		ind[a[i]].pb(i);
	}
	for (int i = 1; i <= n; i++) {
		for (int j = 0; j <= min(k, (int)ind[i].size()) - 1; j++) {
			v.pb(ind[i][j]);
		}
	}
	for (int i = 0; i < (int)v.size() - (int)v.size() % k; i++) {
		col[v[i]] = (i % k) + 1;
	}
	// LET'S CALCULATE FOR EACH OF K COLORS THE NUMBER OF ELEMENTS PAINTED IN THE COLOR — ALL CALCULATED NUMBERS MUST BE EQUAL
	// LET'S ASSERT THIS
	for (int i = 1; i <= n; i++) {
		cout << col[i] << " ";
	}
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}