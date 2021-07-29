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
int n, m, p[N], cn[N];
void solve() {
	cin >> n >> m;
	for (int i = 0; i < n; i++) {
		cin >> p[i];
		p[i]--;
		cn[(i - p[i] + n) % n]++;
	}
	// n / 3 swaps allowed ==> at most 2 n / 3 changed
	// at least n / 3 positions should be fine
	vi v;
	// more swaps ==> less connected components
	for (int i = 0; i < n; i++) if (cn[i] >= n / 3) {
			vi vis(n);
			// running n size loop is ok because cn[i] >= n / 3
			// will appear at most 3 times
			int koi = 0;
			for (int cur = 0; cur < n; cur++) if (vis[cur] == 0) {
					koi++;
					int cur2 = cur;
					while (vis[cur2] != 1) {
						vis[cur2] = 1;
						cur2 = p[(cur2 + i) % n];
						if (vis[cur2])
							break;
					}
				}
			if (koi >= n - m) {
				v.pb(i);
			}
		}
	assert(v.size() <= 3);
	cout << v.size() << " ";
	for (auto &i : v) {
		cout << i << " ";
	}
	fill(cn, cn + n, 0LL);
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}