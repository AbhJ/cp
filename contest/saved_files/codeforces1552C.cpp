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
int n, k, x[N], y[N], not_red[N], ans; vi a, b;

// a A b B

int f(int a, int b, int A, int B) {
	if (a > A)
		swap(a, A),	swap(b, B);
	return a < A and A < b and b < B;
}
void solve() {
	cin >> n >> k;
	a.clear();
	b.clear();
	fill(not_red + 1, not_red + 1 + 2 * n, 0LL);
	for (int i = 1; i <= k; i++) {
		cin >> x[i] >> y[i];
	}
	for (int i = 1; i <= k; i++) {
		if (x[i] > y[i])
			swap(x[i], y[i]);
		not_red[x[i]] = not_red[y[i]] = 1;
		a.pb(x[i]);
		b.pb(y[i]);
	}
	vi red_points;
	for (int i = 1; i <= (n << 1LL); i++) if (!not_red[i]) {
			red_points.pb(i);
		}
	assert((int)red_points.size() == 2 * (n - k));
	for (int i = 0; i < n - k; i++) {
		a.pb(red_points[i]);
		b.pb(red_points[i - k + n]);
	}
	assert((int)b.size() == n);
	assert((int)a.size() == n);
	for (int i = 0; i < n; i++)
		for (int j = i + 1; j < n; j++) {
			ans += f(a[i], b[i], a[j], b[j]);
		}
	cout << ans;
	ans = 0;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}