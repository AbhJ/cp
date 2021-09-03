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
int a[N], n;
vi v;
void default_case() {
	if (a[1] - a[0] < a[n - 1] - a[n - 2]) {
		v.pb(a[0]);
		for (int i = 2; i < n; i++) {
			v.pb(a[i]);
		}
		v.pb(a[1]);
	}
	else {
		v.pb(a[n - 2]);
		for (int i = 0; i < n; i++) {
			if (i != n - 2) {
				v.pb(a[i]);
			}
		}
	}
}
void solve() {
	cin >> n;
	for (int i = 0; i < n; i++) {
		cin >> a[i];
	}
	v.clear();
	sort(a, a + n);
	int mi = min(a[1] - a[0], a[n - 1] - a[n - 2]), pos = inf;
	for (int i = 1; i <= n - 2; i++) {
		if (a[i + 1] - a[i] < mi) {
			mi = a[i + 1] - a[i];
			pos = i;
		}
	}
	if (pos != inf) {
		v.pb(a[pos]);
		for (int i = pos + 2; i < n; i++)
			v.pb(a[i]);
		for (int i = 0; i < pos; i++)
			v.pb(a[i]);
		v.pb(a[pos + 1]);
	}
	else
		default_case();
	for (int i = 0; i < n; i++)
		cout << v[i] << " ";
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}