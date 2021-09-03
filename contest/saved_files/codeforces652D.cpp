#include "bits/stdc++.h"
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
const int N    =     2e6 + 10;
int n, l[N], r[N], cnt, ind[N], ans[N];
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
template<typename T>
using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
static bool cmp(int x, int y) {
	return l[x] < l[y];
}
void solve() {
	cin >> n;
	vpii v;
	ordered_set<int>s;
	for (int i = 1; i <= n; i++) {
		cin >> l[i] >> r[i];
		s.insert(r[i]);
	}
	iota(ind + 1, ind + 1 + n, 1LL);
	sort(ind + 1, ind + 1 + n, cmp);
	for (int pos = 1; pos <= n; pos++) {
		int i = ind[pos];
		// WE ENTER THE INTERVALS WHICH START EARLIER BEFORE ONES STARTING LATER
		// WE DISCARD INTERVALS ONE BY ONE AND COUNT INTERVALS ENDED BEFORE THIS
		// IT WOULD BE COUNT OF CONTAINED / NESTED INTERVALS AS SEEN IN THE CODE
		int contained = s.order_of_key(r[i]);
		s.erase(r[i]);
		ans[i] = contained;
	}
	for (int i = 1; i <= n; i++) {
		cout << ans[i] << "\n";
	}
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	solve(); return 0;
	return 0;
}