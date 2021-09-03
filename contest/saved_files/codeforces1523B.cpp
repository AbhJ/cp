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
#define rep(i, a, b, j) for (int i = a; i <= b; i += j)
#define per(i, b, a) for (int i = b; i >= a; i--)
#define all(x)       x.begin(), x.end()
using namespace std;
const int inf  =     1e18 + 10;
const int N    =     2e6 + 10;
int n, a[N], b[N];
//subtract the previous
//add the next
vector<tuple<int, int, int>> v;
void f(int x, int y, int t) {
	v.pb(x, y, t);
	if (t == 1) {
		b[y] -= b[x];
	}
	else {
		b[x] += b[y];
	}

}
void solve() {
	cin >> n;
	rep(i, 1, n, 1) {
		cin >> a[i];
		b[i] = a[i];
	}
	rep(x, 1, n, 2) {
		int y = x + 1;
		f(x, y, 1);
		f(x, y, 2);
		f(x, y, 1);
		f(x, y, 2);
		f(x, y, 1);
		f(x, y, 2);
	}
	rep(i, 1, n, 1) {
		assert(a[i] == - b[i]);
	}
	assert((int)v.size() <= 5 * n);
	cout << (int)v.size() << "\n";
	for (auto [x, y, z] : v) {
		cout << z << " " << x << " " << y << "\n";
	}
	v.clear();
}
int32_t main() {
	ibs; cti;
	// solve(); return 0;
	int t; cin >> t;
	while (t--) {
		solve();
	}
	return 0;
}