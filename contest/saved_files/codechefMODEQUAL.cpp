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
// WE SHOULD TRY TO REDUCE ALL TO
// 0 <= X <= SMALLEST NUMBER
// OR RATHER {0, SMALLEST NUMBER}
int n, a[N], mi = inf, tot_mi = inf;
void solve() {
	cin >> n;
	for (int i = 1; i <= n; i++) {
		cin >> a[i];
		mi = min(mi, a[i]);
	}
	for (int i = 1; i <= n; i++)
		if (a[i] != mi) {
			tot_mi = min(tot_mi, ((a[i] + 1) >> 1) - 1);
		}
	if (mi > tot_mi)
		cout << n;
	else
		cout << n - count(a + 1, a + 1 + n, mi);
	mi = tot_mi = inf;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}