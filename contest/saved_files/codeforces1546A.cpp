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
const int N    =     1e2 + 10;
int n, a[N], b[N]; vpii v;
void solve() {
	cin >> n; v.clear();
	for (int i = 1; i <= n; i++) {
		cin >> a[i];
	}
	for (int i = 1; i <= n; i++) {
		cin >> b[i];
	}
	for (int i = 1; i < n; i++) {
		for (int j = i + 1; j <= n; j++) {
			while (a[i] < b[i] and a[j] > b[j]) {
				v.pb(j, i);
				a[i]++;
				a[j]--;
			}
		}
		while (a[i] < b[i]) {
			v.pb(n, i);
			a[i]++;
			a[n]--;
		}
		for (int j = i + 1; j <= n; j++) {
			while (a[i] > b[i] and a[j] < b[j]) {
				v.pb(i, j);
				a[j]++;
				a[i]--;
			}
		}
		while (a[i] > b[i]) {
			v.pb(i, n);
			a[n]++;
			a[i]--;
		}
	}
	if (a[n] != b[n]) {
		cout << - 1 << "\n";
		return;
	}
	cout << (int)v.size() << "\n";
	for (auto [x, y] : v) {
		cout << x << " " << y << "\n";
	}
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve();
	}
	return 0;
}