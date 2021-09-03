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
int n, a[N], ev, od, d[N];
int st[N][66], lo[N];

int gc(int l, int r) {
	int j = lo[r - l + 1LL];
	return gcd(st[l][j], st[r - (1 << j) + 1][j]) != 1;
}

void precompute_log() {
	lo[1LL] = 0;
	for (int i = 2; i < N; i++)
		lo[i] = lo[i >> 1LL] + 1LL;
}

void solve() {
	cin >> n;

	for (int i = 0; i < n; i++) {
		cin >> a[i];
		if (i != 0) {
			d[i - 1] = abs(a[i] - a[i - 1]);
			st[i - 1][0] = d[i - 1];
		}
	}

	for (int j = 1; j <= 65; j++)
		for (int i = 0; i + (1LL << j) <= n - 1; i++)
			st[i][j] = gcd(st[i][j - 1], st[i + (1LL << (j - 1))][j - 1]);

	int l, r, ans = 1, m;

	for (int i = 0; i < n - 1; i++) {
		l = i, r = n - 2;
		while (r > l) {
			m = l + r + 1 >> 1;
			if (gc(i, m)) {
				l = m;
			}
			else {
				r = m - 1;
			}
		}
		if (gc(i, l))
			ans = max(ans, l - i + 2LL);
	}

	cout << ans;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	precompute_log();
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}