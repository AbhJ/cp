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
int n, a[N], l, r;
void solve() {
	cin >> n >> l >> r;
	for (int i = 1; i <= n; i++) {
		cin >> a[i];
	}
	sort(a + 1, a + 1 + n);
	int cnt = 0;
	for (int i = 1; i <= n; i++) {
		int lef = lower_bound(a + 1, a + i, l - a[i]) - a;
		int rig = upper_bound(a + 1, a + i, r - a[i]) - a;
		cnt += max(rig - lef, 0LL);
	}
	cout << cnt;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}