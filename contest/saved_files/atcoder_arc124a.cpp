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
const int M    =     998244353;
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
int n, m, k[N], A[N], B[N], ans = 1, fix[N]; char c[N];
void solve() {
	cin >> n >> m;
	for (int i = 1; i <= m; i++) {
		cin >> c[i] >> k[i];
		// ith NUMBER MUST BE k[i]
		if (c[i] == 'L') {
			A[k[i]]++;
		}
		if (c[i] == 'R') {
			B[k[i]]++;
		}
		fix[k[i]] = 1;
		// WE CAN NOT MODIFY k[i]
		// IT HAS THE VALUE i
	}
	for (int i = 1; i <= n; i++) {
		A[i] += A[i - 1];
	}
	for (int i = n; i >= 1; i--) {
		B[i] += B[i + 1];
	}
	for (int i = 1; i <= n; i++) if (fix[i] != 1) {
			ans = mul(ans, add(A[i], B[i]));
		}
	cout << ans;

}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	solve(); return 0;
	return 0;
}