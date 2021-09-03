
/**
 * @author      : abhj
 * @created     : Wednesday Sep 01, 2021 22:20:50 IST
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

int helper (vi& tasks, int sessionTime) {
	int i = 0, j = tasks.size(), koi = 0;
	while (i < j and ++koi) {
		int cnt = tasks[i];
		while (i < j - 1 and cnt + tasks[i + 1] <= sessionTime)
			cnt += tasks[++i];
		while (j - 1 > i and cnt + tasks[j - 1] <= sessionTime)
			cnt += tasks[--j];
		if (j - 1 == i)
			return koi;
		++i;
	}
	return 0;
}

int ans = 1e9, cnt = 0;

int helper_brute (vi& tasks, int sessionTime) {
	vi sess;
	ans = 1e9;
	function<void (vi&, int, vi&, int)> f =
	[&] (vi & tasks, int sessionTime, vi & sess, int pos) {
		cnt++;
		if (cnt > 1e6 or sess.size() > ans){
			return;
		}
		if (pos == tasks.size()) {
			ans = min (ans, (int)sess.size());
			return;
		}
		// ADD TO EXISTING SESSION
		for (int i = 0; i < sess.size(); ++i) {
			if (sess[i] + tasks[pos] <= sessionTime) {
				sess[i] += tasks[pos];
				f (tasks, sessionTime, sess, pos + 1);
				sess[i] -= tasks[pos];
			}
		}
		// CREATE NEW SESSION FOR THIS JOB
		sess.emplace_back (tasks[pos]);
		f (tasks, sessionTime, sess, pos + 1);
		sess.pop_back();
	};
	f (tasks, sessionTime, sess, 0);
	return ans;
}

int minSessions (vi& tasks, int sessionTime) {
	sort (tasks.begin(), tasks.end());
	reverse (tasks.begin(), tasks.end());
	return min (helper (tasks, sessionTime), helper_brute (tasks, sessionTime));
}

int n, x;

void solve() {
	cin >> n >> x;
	vi v (n);
	for (auto &i : v)
		cin >> i;
	cout << minSessions (v, x);
}

int32_t main() {
	ios_base::sync_with_stdio (false);
	cin.tie (0);
	solve();
	return 0;
}
