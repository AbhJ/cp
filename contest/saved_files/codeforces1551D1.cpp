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
	if (n & 1)
		cout << ((((k & 1) == ((m >> 1) & 1)) and (k >= (m >> 1))) ? "YES" : "NO");
	else if (m & 1)
		cout << (((k & 1 ^ 1) and (k <= n * ((m - 1) >> 1))) ? "YES" : "NO");
	else
		cout << ((k & 1 ^ 1) ? "YES" : "NO");
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}