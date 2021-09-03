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
string s, t; int n, vis[N], cnt, koi;
void solve() {
	cnt = koi = 0;
	cin >> n >> s >> t;
	for (int i = 0; i < n; i++) if (t[i] == '1') {
			if (s[i] == '0') {
				s[i] = t[i] = '0';
				vis[i] = 1;
			}
			else if (s[i - 1] == t[i]) {
				s[i - 1] = t[i] = '0';
				vis[i] = 1;
			}
			else if (s[i + 1] == t[i]) {
				s[i + 1] = t[i] = '0';
				vis[i] = 1;
			}
			if (vis[i] == 1) {
				koi++;
				vis[i] = 0;
			}
		}
	cout << koi;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}