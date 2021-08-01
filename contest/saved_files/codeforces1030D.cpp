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
int n, m, k;
void solve() {
	cin >> n >> m >> k;
	if ((2 * n * m) % k != 0) {
		cout << "NO";
		return;
	}
	// THIS ALSO IMPLIES k != 1
	int x = gcd(n, k);
	int a = n / x;
	int b = 2 * m * x / k;
	if (k == x) {
		a *= 2;
		b /= 2;
	}
	cout << "YES\n" << "0 0\n" << a << " 0\n0 " << b;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	solve();
	return 0;
}