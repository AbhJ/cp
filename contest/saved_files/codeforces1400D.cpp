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
const int N    =     3e3 + 10;
int n, a[N], ans, num_of_i_s[N];
void solve() {
	cin >> n;
	for (int i = 1; i <= n; i++)
		cin >> a[i];
	for (int i = 0; i <= n + 1; i++)
		num_of_i_s[i] = 0;
	for (int j = 1; j <= n; j++) {
		num_of_i_s[a[j - 1]]++;
		int ik_pairs = 0;
		for (int k = j + 1; k <= n; k++) {
			if (a[j] == a[k])
				// jl pairs formed
				ans += ik_pairs;
			ik_pairs += num_of_i_s[a[k]];
		}
	}
	cout << ans;
	ans = 0;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}