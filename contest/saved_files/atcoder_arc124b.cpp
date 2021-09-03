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
int n, a[N], b[N];
void solve() {
	// x CAN TAKE ATMOST ONE OF n VALUES
	cin >> n;
	for (int i = 1; i <= n; i++) {
		cin >> a[i];
	}
	vi v, B, tmp, fin;
	// HAS ALL OF POSSIBLE x VALUES
	for (int i = 1; i <= n; i++) {
		cin >> b[i];
		v.pb(b[i] ^ a[1]);
		B.pb(b[i]);
	}
	sort(B.begin(), B.end());
	sort(v.begin(), v.end());
	v.erase(unique(v.begin(), v.end()), v.end());
	for (auto &x : v) {
		tmp.clear();
		for (int i = 1; i <= n; i++) {
			tmp.pb(a[i] ^ x);
		}
		sort(tmp.begin(), tmp.end());
		if (tmp == B)
			fin.pb(x);
	}
	cout << fin.size() << "\n";
	for (auto &i : fin) {
		cout << i << " ";
	}
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	solve(); return 0;
	return 0;
}