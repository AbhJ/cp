
/**
 * @author      : abhj
 * @created     : Saturday Sep 11, 2021 18:08:42 IST
 * @filename    : f.cpp
 */

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
const int N    =     4e2 + 10;
const int NN    =    2e6 + 10;
const int M    =     998244353;
int add (int a, int b) {
	a += b;
	return (a >= M ? a - M : a);
}
int mul (int a, int b) {
	return (a * b) % M;
}
int sub (int a, int b) {
	return (a - b + M) % M;
}
int powM (int b, int p) {
	int r = 1;
	for (; p; b = mul (b, b), p >>= 1)
		if (p & 1)
			r = mul (r, b);
	return r;
}
int invM (int x) {
	return powM (x, M - 2);
}
int f[NN], b[NN];
void binomialCoeff() {
	f[0] = 1;
	for (int i = 1; i <= NN - 1; i++)
		f[i] = mul (f[i - 1], i);
	b[NN - 1] = powM (f[NN - 1], M - 2);
	for (int i = NN - 2; i >= 0; i--)
		b[i] = mul (b[i + 1], i + 1);
}
int C (int n, int r) {
	return (r > n ? 0LL : mul (f[n], mul (b[r], b[n - r])));
}
// INTIALLY WE CAN ONLY REMOVE ITEMS ADJACENT NUMBERED
int n, m, dp[N][N], ok[N][N], A[NN], B[NN], l, r;

void solve() {
	cin >> n >> m;
	memset (dp, -1, sizeof dp);
	set<int> s_a, s_b;
	for (int i = 0; i < m; i++) {
		cin >> A[i] >> B[i];
		ok[A[i]][B[i]] = ok[B[i]][A[i]] = 1;
	}
	function <int (int, int)> f = [&] (int l, int r) -> int {
		if (l > r)
			return 1;
		if (dp[l][r] != - 1)
			return dp[l][r];
		int ans = 0;
		for (int i = l + 1; i <= r; i++)
			if (ok[l][i] and (i - 1 - l - 1 + 1) & 1 ^ 1 and (r - i - 1 + 1) & 1 ^ 1) {
				// l, ..., i, ..., r
				int countLeft = (i - 1 - l - 1 + 1) / 2;
				// PAIRS IN LEFT OF i
				int countRight = (r - i - 1 + 1) / 2;
				// PAIRS IN RIGHT OF i
				ans = add (ans, mul (C (countLeft + countRight + 1, countLeft + 1), mul (f (l + 1, i - 1), f (i + 1, r))));
			}
		return dp[l][r] = ans;
	};
	cout << f (1, n << 1);
}

int32_t main() {
	ios_base::sync_with_stdio (false);
	cin.tie (0);
	binomialCoeff();
	solve();
	return 0;
}
