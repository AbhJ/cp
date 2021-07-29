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
const int N    =     1e5 + 10;
int n, a[N];
void solve() {
	cin >> n;
	vvi adj(N);
	for (int i = 0; i < n; ++i)
		cin >> a[i], adj[a[i]].pb(i);
	int k = 0, dif = 0;
	for (int i = 0; i < N; ++i) {
		int bam = 0, dan = 0;
		for (auto j : adj[i])
			bam += (abs(j - k) & 1 ^ 1), dan += (abs(j - k) & 1);
		while (bam or dan)
			if (dif & 1 ^ 1) {
				if (!bam) {
					cout << "NO";
					return;
				}
				bam--, dif++;
			}
			else {
				if (!dan) {
					cout << "NO";
					return;
				}
				dan--, dif++;
			}
	}
	cout << "YES";
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}