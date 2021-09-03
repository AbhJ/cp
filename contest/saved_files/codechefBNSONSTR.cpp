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
int n, ans = inf; string s;
vector<int> fact(int n) {
	vector<int> v;
	for (int i = 1; i <= sqrt(n); ++i) {
		if (n % i == 0) {
			v.push_back(i);
			if (i * i != n) {
				v.push_back(n / i);
			}
		}
	}
	return v;
}
void solve() {
	cin >> n >> s;
	int cn0 = 0, cn1 = 0;
	for (int i = 0; i < n; i++) {
		cn1 += (s[i] == '1');
		cn0 += (s[i] == '0');
	}
	vi v = fact(n);
	for (auto I : v) {
		vi ma;
		for (int i = 0; i < I; i++) {
			ma.pb(0);
			for (int j = i; j < n; j += I) {
				if (s[j] == '1') {
					ma[i]++;
				}
			}
		}
		for (int i = 0; i < I; i++)
			ans = min(ans, cn1 - 2 * ma[i] + n / I );
	}
	cout << ans;
	ans = inf;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}