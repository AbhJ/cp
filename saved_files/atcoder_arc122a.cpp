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
const int N    =     1e5 + 10;
const int M    =     1e9 + 7;
int add(int a, int b) {
	a += b;
	return (a >= M ? a - M : a);
}
int mul(int a, int b) {
	return (a * b) % M;
}
int sub(int a, int b) {
	// return a - b;
	return (a - b + M) % M;
}
int powM(int b, int p) {
	int r = 1;
	for (; p; b = mul(b, b), p >>= 1) {
		if (p & 1)
			r = mul(r, b);
	}
	return r;
}
int invM(int x) {
	return powM(x, M - 2);
}
int f[N], b[N];
void binomialCoeff() {
	f[0] = 1;
	for (int i = 1; i <= N - 1; i++) {
		f[i] = mul(f[i - 1], i);
	}
	b[N - 1] = powM(f[N - 1], M - 2);
	for (int i = N - 2; i >= 0; i--) {
		b[i] = mul(b[i + 1], i + 1);
	}
}
int C(int n, int r) {
	return (r > n ? 0LL : (f[n] * mul(b[r], b[n - r]) % M) % M);
}
int n, a[N], dp[2][N], tmp[2][N], cnt;
void solve() {
	cin >> n;
	dp[0][1] = 1;
	dp[1][1] = 0;
	for (int i = 1; i <= n; i++) {
		cin >> a[i];
	}
	tmp[0][1] = a[1]; tmp[1][1] = 0;
	for (int i = 2; i <= n; i++) {
		// README
		// add(tmp[0][i - 1], tmp[1][i - 1]) is sum created by previous formula
		// mul(dp[0][i], a[i]) is sum contributed by current member
		dp[0][i] = add(dp[0][i - 1], dp[1][i - 1]);
		dp[1][i] = dp[0][i - 1];
		tmp[0][i] = add(add(tmp[0][i - 1], tmp[1][i - 1]), mul(dp[0][i], a[i]));
		tmp[1][i] = sub(tmp[0][i - 1], mul(dp[1][i], a[i]));
	}
	cout << add(tmp[1][n], tmp[0][n]);
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	solve(); return 0;
	return 0;
}