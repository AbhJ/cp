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
int n, taka;
long double a[N], l, r, ans = inf, m1, m2;
long double f(long double x) {
	long double cost = 0;
	for (int i = 1; i <= n; i++) {
		cost += max(x, a[i] - x);
	}
	ans = min(ans, cost);
	return cost;
}
void solve() {
	cin >> n;
	for (int i = 1; i <= n; i++) {
		cin >> a[i];
	}
	r = inf;
	while (l / (long double) n <= r / (long double) n - 1e-6) {
		m1 = (r - l) / 3.0 + l;
		m2 = r - (r - l) / 3.0;
		if (f(m1) < f(m2))
			r = m2;
		else
			l = m1;
	}
	cout << fixed << setprecision(6) << ans / (long double) n;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	solve(); return 0;
	return 0;
}