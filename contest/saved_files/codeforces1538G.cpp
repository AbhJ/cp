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
const int inf  =     1e9 + 10;
const int N    =     2e6 + 10;
int x, y, a, b, gif1, gif2, gif3;
int f(int m) {
	int v1 = ceil((y - m * a) * 1.0 / (b - a));
	int v2 = floor((x - m * b) * 1.0 / (a - b));
	// k is less than equal to n
	// if (v1 > v2)
	// 	swap(v1, v2);
	v1 = max(v1, 0LL);
	v2 = min(v2, m);
	return v1 <= v2;
}
void solve() {
	cin >> x >> y >> a >> b;
	// x >= k * a + (n - k) * b
	// x >= (a - b) * k + n * b
	// y >= (b - a) * k + n * a


	// x + y - n * (a + b) >= 0


	// x - y >= 2 * (a - b) * k + n * (b - a)
	// (x - y + n * (a - b)) / 2 * (a - b) >= k
	if (a < b)
		swap(a, b);
	if (a == b) {
		cout << min(x, y) / a;
		return;
	}
	int l = 0, r = inf, m;
	while (r > l) {
		m = l + r + 1 >> 1;
		if (f(m))
			l = m;
		else
			r = m - 1;
	}
	if (f(l + 1))
		cout << l + 1;
	else if (f(l))
		cout << l;
	else if (f(l - 1))
		cout << l - 1;
	else
		cout << 0;
	// cout << (x - y + n * (a - b)) / 2 * (a - b);
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}