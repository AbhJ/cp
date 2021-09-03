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
#define rep(i, a, b) for (int i = a; i <= b; i++)
#define per(i, b, a) for (int i = b; i >= a; i--)
#define all(x)       x.begin(), x.end()
using namespace std;
const int inf  =     1e18 + 10;
const int N    =     2e6 + 10;
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
	rep(i, 1, N - 1) {
		f[i] = mul(f[i - 1], i);
	}
	b[N - 1] = powM(f[N - 1], M - 2);
	per(i, N - 2, 0) {
		b[i] = mul(b[i + 1], i + 1);
	}
}
int C(int n, int r) {
	return (r > n ? 0LL : (f[n] * mul(b[r], b[n - r]) % M) % M);
}
void solve() {
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	// solve(); return 0;
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}