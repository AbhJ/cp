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
int n, a[N], l, r, su, cn, k, mi = inf;
void solve() {
	cin >> n;
	for (int i = 1; i <= 2 * n - 1; i++) {
		cin >> a[i];
	}
	for (int i = 1; i <= 2 * n - 1; i++) {
		if (a[i] < 0) {
			k++;
			a[i] = - a[i];
		}
		mi = min(mi, a[i]);
	}
	su = accumulate(a + 1, a + 2 * n, 0LL);
	if (n & 1 or k & 1 ^ 1) {
		cout << su;
		return;
	}
	su += - 2 * mi;
	cout << su;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	solve(); return 0;
	return 0;
}