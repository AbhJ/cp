

/**
 * @author      : abhj
 * @created     : Tuesday Aug 24, 2021 02:12:08 IST
 * @filename    : a.cpp
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

int n, ans, tmp, koi;
string s[N], S, T;

int f(string S) {
	for(int i = 0; i < n; i++)
		if(s[i] == S)
			return 1;
	return 0;
}

int ok(char ty) {
	S.clear();
	for(int i = 0; i < n; i++) {
		int z = 0, o = 0;
		for(int j = 0; j < n; j++) {
			z += s[i][j] != ty;
			o += s[i][j] == ty;
		}
		if(o > z) {
			S += '0' + ((ty - '0') ^ 1);
		}
		else {
			S += '0';
		}
	}
	return f(S) == 0;
}

void solve() {
	cin >> n;
	for(int i = 0; i < n; i++)
		cin >> s[i];
	S = string(n, '0');
	T = string(n, '1');
	if(f(S) == 0)
		cout << S;
	else if (f(T) == 0)
		cout << T;
	else if(ok('0'))
		cout << S;
	else if(ok('1'))
		cout << S;
	else assert(1);
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
