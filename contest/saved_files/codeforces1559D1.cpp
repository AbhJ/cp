
/**
 * @author      : abhj
 * @created     : Sunday Aug 15, 2021 20:06:40 IST
 * @filename    : a.cpp
 */

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
struct dsu {
public:
	dsu() : _n(0) {}
	explicit dsu(int n) : _n(n), parent_or_size(n, -1) {}

	int merge(int a, int b) {
		assert(0 <= a && a < _n);
		assert(0 <= b && b < _n);
		int x = leader(a), y = leader(b);
		if (x == y) return x;
		if (-parent_or_size[x] < -parent_or_size[y]) std::swap(x, y);
		parent_or_size[x] += parent_or_size[y];
		parent_or_size[y] = x;
		return x;
	}

	bool same(int a, int b) {
		assert(0 <= a && a < _n);
		assert(0 <= b && b < _n);
		return leader(a) == leader(b);
	}

	int leader(int a) {
		assert(0 <= a && a < _n);
		if (parent_or_size[a] < 0) return a;
		return parent_or_size[a] = leader(parent_or_size[a]);
	}

	int size(int a) {
		assert(0 <= a && a < _n);
		return -parent_or_size[leader(a)];
	}

	std::vector<std::vector<int>> groups() {
		std::vector<int> leader_buf(_n), group_size(_n);
		for (int i = 0; i < _n; i++) {
			leader_buf[i] = leader(i);
			group_size[leader_buf[i]]++;
		}
		std::vector<std::vector<int>> result(_n);
		for (int i = 0; i < _n; i++) {
			result[i].reserve(group_size[i]);
		}
		for (int i = 0; i < _n; i++) {
			result[leader_buf[i]].push_back(i);
		}
		result.erase(
		    std::remove_if(result.begin(), result.end(),
		[&](const std::vector<int>& v) { return v.empty(); }),
		result.end());
		return result;
	}

private:
	int _n;
	// root node: -1 * component size
	// otherwise: parent
	std::vector<int> parent_or_size;
};

int x, y, n, m1, m2, cnt;
vi a1[N], a2[N];
void solve() {
	cin >> n >> m1 >> m2;
	dsu dsu1(n);
	dsu dsu2(n);
	for (int i = 0; i < m1; i++) {
		cin >> x >> y;
		a1[x].pb(y);
		a1[y].pb(x);
		dsu1.merge(y - 1, x - 1);
	}
	for (int i = 0; i < m2; i++) {
		cin >> x >> y;
		a2[x].pb(y);
		a2[y].pb(x);
		dsu2.merge(y - 1, x - 1);
	}
	vpii v;
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			if (dsu1.same(i, j) == 0 and dsu2.same(i, j) == 0) {
				cnt++;
				dsu1.merge(i, j);
				dsu2.merge(i, j);
				v.pb(i, j);
			}
	cout << cnt << "\n";
	for (auto [x, y] : v) {
		cout << x + 1 << " " << y + 1 << "\n";
	}
}

int32_t main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	solve();
	return 0;
}
