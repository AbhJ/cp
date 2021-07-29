//Coded by Abhijay Mitra (AbJ)
#include "bits/stdc++.h"
#define double       long double
#define int          long long int
#define ibs          ios_base::sync_with_stdio(false)
#define cti          cin.tie(0)
#define mp           make_pair
#define pb           emplace_back
#define F            first
#define S            second
using vi       =     std::vector<int>;
using vvi      =     std::vector<vi>;
using pii      =     std::pair<int, int>;
using vpii     =     std::vector<pii>;
using vvpii    =     std::vector<vpii>;
#define rep(i, a, b) for (int i = a; i <= b; i++)
#define per(i, b, a) for (int i = b; i >= a; i--)
#define all(x)       x.begin(), x.end()
using namespace std;
const int inf  =     1e18 + 10;
const int N    =     2e6 + 10;
int n, a[N], c[N], vis[N], A, C, pos, ans;
set<tuple<int, int, int>>s;
tuple<int, int, int>f(int A, int C) {
	if (s.lower_bound(tuple<int, int, int> {A + C, - inf, - inf}) != s.begin())
		return *s.lower_bound(tuple<int, int, int> {A + C, - inf, - inf});
	return *s.begin();
}
void solve() {
	cin >> n;
	// max(c[i], a[j] - a[i]) = c[i] + max(0, a[j] - a[i] - c[i]);
	set<pii>S;
	vpii v;
	rep(i, 1, n) {
		cin >> a[i] >> c[i];
		v.pb(a[i], c[i]);
		S.emplace(a[i], c[i]);
		ans += c[i];
	}
	sort(all(v));
	priority_queue<pii>q;
	q.emplace(0, 1);
	while (q.empty() == 0) {
		auto [taka, i] = q.top();
		q.pop();
		if (vis[i] == 1)
			continue;
		vis[i] = 1;
		if (i == n) {
			cout << ans + (- taka);
			return;
		}
		if (i != 1) {
			q.emplace(taka, i - 1);
		}
		q.emplace(taka, lower_bound(all(v), mp(v[i - 1].F + v[i - 1].S, inf)) - v.begin());
		int cnt = lower_bound(all(v), mp(v[i - 1].F + v[i - 1].S, inf)) - v.begin();
		if (cnt != n)
			q.emplace(taka - (+ v[cnt].F - v[i - 1].F - v[i - 1].S), cnt + 1);
	}
	cout << ans;
}
int32_t main() {
	ibs; cti;
	solve(); return 0;
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}