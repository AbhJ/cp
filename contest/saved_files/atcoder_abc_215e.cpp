
/**
 * @author      : abhj
 * @created     : Tuesday Aug 24, 2021 11:25:21 IST
 * @filename    : e.cpp
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
const int N    =     1e3 + 10;
const int M    =     998244353;
int n, dp[N][1 << 10][10], cnt;
string s;
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
void solve() {
	cin >> n >> s;
	for(int i = 0; i < n; i++) {
		//IF THE CHARACHARTER IS NOT NEW
		for(int mas = 0; mas < (1 << 10); mas++) {
			for(int j = 0; j < 10; j++) if((mas >> j) & 1) {
					char exp = 'A' + j;
					if(exp != s[i])
						dp[i][mas][j] = (i == 0 ? 0 : dp[i - 1][mas][j]);
					else
						dp[i][mas][j] = mul(2, (i == 0 ? 0 : dp[i - 1][mas][j]));
					//THE LAST CHARACTER USED IS EXP
					//NOW WE ARE PLACING s[i]
					//NUMBER OF WAYS IS dp[i][j]
					//TAKING AND NOT TAKING
				};
		}
		//IF THE CHARACTER IS NEW
		for(int mas = 0; mas < (1 << 10); mas++) {
			for(int j = 0; j < 10; j++) if((mas >> (s[i] - 'A')) & 1 ^ 1) {
					dp[i][mas | (1 << (s[i] - 'A'))][s[i] - 'A'] = add(
								(i == 0 ? 0 : dp[i - 1][mas][j]),
								dp[i][mas | (1 << (s[i] - 'A'))][s[i] - 'A']
							);
				}
		}
		dp[i][1 << (s[i] - 'A')][s[i] - 'A'] = 	add(1, dp[i][1 << (s[i] - 'A')][s[i] - 'A']);
	}
	for(int mas = 0; mas < (1 << 10); mas++)
		for(int j = 0; j < 10; j++)
			cnt = add(dp[n - 1][mas][j], cnt);
	cout << cnt;
}
int32_t main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	solve();
	return 0;
}
