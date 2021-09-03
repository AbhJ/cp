
/**
 * @author      : abhj
 * @created     : Monday Aug 02, 2021 18:58:16 IST
 * @filename    : d.cpp
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
int n, a[N], p[N];

vi handle_ones() {
	vi v;

	for(int i = 1; i <= n; i++) {
		if(a[i] == 1)
			v.pb(a[i]);
	}

	for(int i = 1; i <= n; i++) {
		if(p[a[i] + 1] and v.empty() == 0 and a[i] != 1) {
			v.pb(a[i]);
			break;
		}
	}

	if(v.empty() == 1)
		v.pb(a[n]);

	return v;
}

void solve() {
	cin >> n;

	for(int i = 1; i <= n; i++) {
		cin >> a[i];
	}

	fill(p + 2, p + N, 1LL);

	for (int i = 2; i <= N - 1; i++) {
		if (p[i] == 1) {
			for (int j = (i * i); j < N; j += i) {
				p[j] = 0;
			}
		}
	}

	// ONES CAN BE CLUBBED WITH ONLY SOME EVEN NUMBERS
	// HANDLE THAT CASE DIFFERENTLY

	vi ans = handle_ones();

	// IF ONES ARE NOT PRESENT, WE CAN HANDLE CASE USING
	// ONLY TWO INTEGERS, ONE ODD AND ONE EVEN TO AVOID
	// EVEN PARITY.
	// SINCE BOTH OF THEM CANNOT BE ONE, THE SUM CAN NOT
	// BE TWO.

	for(int i = 1; i <= n; i++) {
		for(int j = i; j <= n; j++) {
			if(((a[i] ^ a[j]) & 1LL) == 1 and (ans.size() < 2) and p[a[i] + a[j]] == 1) {
				ans = vi {a[i], a[j]};
			}
		}
	}

	cout << ans.size() << "\n";
	for(auto &i: ans) {
		cout << i << " ";
	}
}

int32_t main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	solve();
	return 0;
}
