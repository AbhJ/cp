

/**
 * @author      : abhj
 * @created     : Sunday Aug 22, 2021 09:59:10 IST
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

int n;
vector<double> printer_vec(N);

// idea present in https://math.stackexchange.com/questions/451558/how-to-find-the-sum-of-the-series-1-frac12-frac13-frac14-dots
void solve() {
	cin >> n;
	if (n < N)
		cout << printer_vec[n];
	else
		cout << 0.5772156649015328606 + log(n) + (double)1 / (double)(n << 1LL);
}

int32_t main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	printer_vec[1] = 1;
	cout << fixed << setprecision(7);
	for (double i = 2; i < N; i++) {
		printer_vec[i] = printer_vec[i - 1] + (double)1 / i;
	}
	int t, x = 0;
	cin >> t;
	while (t--) {
		cout << "Case #" << ++x << ": ";
		solve();
		cout << "\n";
	}
	return 0;
}
