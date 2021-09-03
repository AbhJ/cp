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
int n, a[N], b[N], ans; map<int, int> m; queue<int> q;
int LIS(vi& vec) {
	int n = vec.size();
	int ans = 0;
	fill(b, b + n, 1LL);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < i; j++) {
			if (vec[i] >= vec[j]) {
				b[i] = max(b[i], 1 + b[j]);
				ans = max(ans, b[i]);
			}
		}
	}
	return ans;
}

bool g(vector<int> &vec) {
	int L1 = LIS(vec);
	reverse(vec.begin(), vec.end());
	int L2 = LIS(vec);
	return max(L1, L2) <= 2;
}


bool f(queue<int> q) {
	vi vec;
	while (!q.empty()) vec.pb(q.front()), q.pop();
	return g(vec);
}
int i, j;
void solve() {
	cin >> n; q = queue<int>();
	for (int i = 0; i < n; i++) {
		cin >> a[i];
	}
	ans = 0;
	i = 0, j = 0;
	while (j < n) {
		q.push(a[j]);
		while (f(q) == 0)
			ans += j - i++, q.pop();
		j++;
	}
	cout << ans + (((j - i) * ((j - i) + 1)) >> 1LL);
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}