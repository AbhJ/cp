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
vi a; int n, q;
void solve() {
	cin >> n >> q;
	a = vi(n);
	for (auto &i : a)
		cin >> i;
	sort(a.begin(), a.end());
	while (q--) {
		int x, beat;
		cin >> x;
		auto it = lower_bound(a.begin(), a.end(), x);
		if (it == a.end()) {
			beat = n;
		}
		else if (*it == x) {
			cout << 0 << "\n";
			continue;
		}
		else beat = (it - a.begin()) & 1 ^ 1;
		cout << (beat ? "POSITIVE\n" : "NEGATIVE\n");
	}
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	solve(); return 0;
	return 0;
}