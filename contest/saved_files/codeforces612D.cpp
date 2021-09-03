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
int n, k, l[N], r[N], koi, curR, curL = - inf;
void solve() {
	cin >> n >> k;
	vpii v, V;
	for (int i = 1; i <= n; i++) {
		cin >> l[i] >> r[i];
		v.pb(l[i], 0);
		v.pb(r[i], 1);
	}
	sort(v.begin(), v.end());
	for (auto &i : v) {
		if (i.S == 1) {
			koi--;
			if (koi == k - 1) {
				curR = i.F;
				assert(curL <= curR and curL != - inf);
				if (V.empty() == 0) {
					auto [lasL, lasR] = V.back();
					if (lasR == curL) {
						V.pop_back();
						V.pb(lasL, curR);
						continue; ==
					}
				}
				V.pb(curL, curR);
			}
		}
		if (i.S == 0) {
			koi++;
			if (koi == k) {
				curL = i.F;
			}
		}
	}
	cout << (int)V.size() << "\n";
	for (auto &[x, y] : V) {
		cout << x << " " << y << "\n";
	}
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	solve(); return 0;
	return 0;
}