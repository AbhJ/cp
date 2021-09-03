
/**
 * @author      : abhj
 * @created     : Tuesday Aug 17, 2021 11:00:39 IST
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
const int N    =     1e5 + 10;

int n, cnt;
vvi prime_vector_sieve() {
	vvi prime(N);
	for (int i = 2; i < N; ++i)if (prime[i].empty()) {
			for (int j = i; j < N; j += i) {
				prime[j].pb(i);
			}
		}
	return prime;
}

vi fact(int n) {
	vi v;
	for (int i = 1; i * i <= n; ++i) {
		if (n % i == 0) {
			v.pb(i);
			if (i * i != n) {
				v.pb(n / i);
			}
		}
	}
	return v;
}

vi __fact[N];

void solve() {
	cin >> n;

//  A = Bx + C
//  (N - C) / B + 1 possible A s are there

	cnt = 0;

	for(int b = 1; b <= n; b++) {
		vi &fac = __fact[b];
		for(auto c: fac) if (c < b) {
				cnt += (n - c) / b + 1;
			}
	}

	cout << cnt;
}

int32_t main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	
	for(int i = 1; i < N; i++){
		__fact[i] = fact(i);
	}

	int t;
	cin >> t;
	while(t--)
		solve(), cout << "\n";
	return 0;
}
