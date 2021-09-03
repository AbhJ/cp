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
const int N    =     1e6 + 10;
int n, a[N], st[N][65];
int gcd(int L, int R) {
	int j = log2(R - L + 1LL);
	int g = __gcd(st[L][j], st[R - (1LL << j) + 1LL][j]);
	return g;
}
void solve() {
	cin >> n;
	for (int i = 0; i < n; i++) {
		cin >> a[i];
		a[i + n] = a[i];
	}
	for (int i = 0; i < 2 * n; i++)
		st[i][0] = a[i];
	for (int j = 1; j <= 64; j++)
		for (int i = 0; i + (1LL << j) <= 2 * n; i++)
			st[i][j] = __gcd(st[i][j - 1LL], st[i + (1LL << (j - 1LL))][j - 1LL]);
	int l = 0, r = n, m;
	while (r > l) {
		m = l + r >> 1LL;
		int F = 0, las = - 1;
		for (int i = 0; i < n; i++) {
			int cur_g = gcd(i, i + m);
			if (las == - 1) {
				las = cur_g;
			}
			else if (las != cur_g) {
				F = 1;
				break;
			}
		}
		if (F == 1)
			l = m + 1;
		else
			r = m;
	}
	cout << r;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}