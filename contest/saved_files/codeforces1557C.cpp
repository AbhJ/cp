/**
 * @author      : abhj
 * @created     : Monday Aug 16, 2021 11:09:42 IST
 * @filename    : c.cpp
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
const int N    =     2e6 + 10;
int n, k, a[N];
const int M    =     1e9 + 7;
int add(int a, int b) {
	a += b;
	return (a >= M ? a - M : a);
}
int mul(int a, int b) {
	return (a * b) % M;
}
int sub(int a, int b) {
	return (a - b + M) % M;
}
int powM(int b, int p) {
	int r = 1;
	for (; p; b = mul(b, b), p >>= 1)
		if (p & 1)
			r = mul(r, b);
	return r;
}
int invM(int x) {
	return powM(x, M - 2);
}
int f[N], b[N];
void binomialCoeff() {
	f[0] = 1;
	for (int i = 1; i <= N - 1; i++)
		f[i] = mul(f[i - 1], i);
	b[N - 1] = powM(f[N - 1], M - 2);
	for (int i = N - 2; i >= 0; i--)
		b[i] = mul(b[i + 1], i + 1);
}
int C(int n, int r) {
	return (r > n ? 0LL : mul(f[n], mul(b[r], b[n - r])));
}
int dp[N][3];
const int _equal = 1, _nequal = 2;
void solve() {
	cin >> n >> k;

	// if n is even, we want a bit set on all

	// if n is odd, we want a bit set on all before a situation
	// comes with n odd set bits

	// if n is even
	// 		if number of set bits is n, 	and = 1 and xor = 0 -- WIN (_nequal)
	// 		if number of set bits is odd, 	and = 0 and xor = 1 -- LOSS
	// 		if number of set bits is even,	and = 0 and xor = 0 -- WIN (_equal)

	// if n is odd
	// 		if number of set bits is n, 	and = 1 and xor = 1 -- WIN (_equal)
	// 		if number of set bits is even, 	and = 0 and xor = 0 -- WIN (_equal)
	// 		if number of set bits is odd,	and = 0 and xor = 1 -- LOSS
	dp[0][_equal] = dp[0][_nequal] = 1;
	if (n & 1 ^ 1)
		for (int i = 1; i <= k; i++) {
			dp[i][_nequal] = mul(powM(2, n), dp[i - 1][_nequal]);
			// _nequal remains _nequal on choosing all bits

			dp[i][_equal] = add(mul(powM(2, n - 1) - 1, dp[i - 1][_equal]), mul(1, dp[i - 1][_nequal]));
			// _equal composed from _nequal and _equal
		}
	if (n & 1)
		for (int i = 1; i <= k; i++) {
			dp[i][_nequal] = mul(powM(2, n), dp[i - 1][_nequal]);
			// already won, dont see way to convert _equal to _nequal here

			dp[i][_equal] = mul(1 + powM(2, n - 1), dp[i - 1][_equal]);
			// all bits set or even subset set
			// https://math.stackexchange.com/questions/1338787/number-of-subsets-with-even-number-of-elements
		}
	cout << add(dp[k][_equal], 0);
}

int32_t main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	int t;
	cin >> t;
	while (t--) {
		solve();
		cout << "\n";
	}
	return 0;
}
