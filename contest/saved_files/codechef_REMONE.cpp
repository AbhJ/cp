

/**
 * @author      : abhj
 * @created     : Sunday Aug 29, 2021 17:33:42 IST
 * @filename    : b.cpp
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

int n, a[N], b[N], x;

void solve() {
	cin >> n;
	for (int i = 0; i < n; i++)
		cin >> a[i];
	for (int i = 0; i < n - 1; i++)
		cin >> b[i];
	int biggest_b = *max_element (b, b + n - 1);
	for (int i = 0; i < n; i++) {
		if (a[i] > biggest_b)
			a[i] = 0;
	}
	int biggest_a = *max_element (a, a + n);
	cout << biggest_b - biggest_a;
}

int32_t main() {
	ios_base::sync_with_stdio (false);
	cin.tie (0);
	int t;
	cin >> t;
	while (t--) {
		solve();
		cout << "\n";
	}
	return 0;
}
