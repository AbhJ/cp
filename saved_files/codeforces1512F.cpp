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
int n, a[N], b[N], l, r, m1, m2, c;
int f(int cour) {
	int day = cour + 1;
	if (day > n or cour < 0)
		return inf;
	int tot_days = 0, cur_taka = 0;
	for (int i = 1; i <= cour; i++) {
		int cyc = (b[i] + a[i] - 1 - cur_taka) / a[i];
		tot_days += cyc + 1;
		cur_taka += a[i] * cyc - b[i];
		if (cyc < 0)
			return inf;
	}
	return tot_days + max((c + a[day] - 1 - cur_taka) / a[day], 0LL);
}
int g() {
	int tot_days = 0, cur_taka = 0;
	int ans = inf;
	for (int i = 1; i <= n; i++) {
		ans = min(ans, max(tot_days, tot_days + (c + a[i] - 1 - cur_taka) / a[i]));
		int cyc = (b[i] + a[i] - 1 - cur_taka) / a[i];
		tot_days += cyc + 1;
		cur_taka += a[i] * cyc - b[i];
	}
	return ans;
}
void solve() {
	cin >> n >> c;
	for (int i = 1; i <= n; i++) {
		cin >> a[i];
	}
	for (int i = 1; i <= n - 1; i++) {
		cin >> b[i];
	}
	l = 0, r = n - 1;
	// m is the last course taken, m + 1 is current position
	while (r > l + 2) {
		m1 = l + (r - l) / 3;
		m2 = r - (r - l) / 3;
		if (f(m1) < f(m2))
			r = m2;
		else
			l = m1;
	}
	cout << min({f(l), f(r), f(l + 1), g()});
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}