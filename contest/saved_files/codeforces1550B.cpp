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
int n, a, b, koi0, koi1; string s;

void solve() {
	cin >> n >> a >> b >> s;
	if (b >= 0) {
		cout << (a + b) * n;
		return;
	}
	for (int i = 1; i < n; i++) {
		koi0 += (s[i] == '0' and s[i - 1] == '1');
		koi1 += (s[i] == '1' and s[i - 1] == '0');
	}
	koi0 += (s[0] == '0');
	koi1 += (s[0] == '1');
	cout << a * n + b * ((min(koi0, koi1)) + 1LL);
	koi0 = koi1 = 0;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}