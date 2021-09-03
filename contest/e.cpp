
/**
 * @author      : abhj
 * @created     : Thursday Sep 02, 2021 23:22:21 IST
 * @filename    : e.cpp
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

int n, k, a[N], ans;

void brute() {
	priority_queue<int>q;
	for (int i = 0; i < n; i++)
		q.emplace (a[i]);
	int ans = 0;
	while (k--) {
		ans += max (0LL, q.top());
		int x = q.top();
		q.pop();
		q.emplace (x - 1);
	}
	cout << ans;
}

void solve() {
	cin >> n >> k;
	for (int i = 0; i < n; i++)
		cin >> a[i];
	if (k < 2e6) {
		brute();
		return;
	}
	sort (a, a + n, greater<int>());
	for (int i = 0; i < n; i++)
		if (k != 0) {
			int dif = a[i] - a[i + 1];
			int num = i + 1;
			/*
			 * NUM MEANS HOW MANY ELEMETS <= i HAVE BEEN CONVERTED TO
			 * VALUE EQUAL TO a[i]
			 * EACH REACHES a[i] - dif, SO LAST CONTRIBUTON
			 * FROM EACH IS a[i] - dif + 1
			 */
			int taka = 2 * a[i + 1] + dif + 1;
			taka *= dif;
			taka >>= 1;
			taka *= num;
			int moves = num * dif;
			if (moves <= k) {
				/*
				 * EASY CASE
				 * WE REDUCE ALL THESE NUMBERS
				 */
				k -= moves;
			}
			else {
				// HOW MUCH WILL EACH PERSON DECREASE ?
				int each_dif = k / num;
				// HOW MANY REST WILL DECREASE BY ONE
				int how_many = k % num;
				/*
				 * EACH REACHES a[i] - each_dif, SO LAST CONTRIBUTON
				 * FROM EACH IS a[i] - each_dif + 1
				 */
				taka = (2 * a[i] - each_dif + 1) * each_dif * num;
				taka >>= 1;
				/*
				 * AFTER EACH IS DECREASED BY each_dif, EACH ELEMENT WHICH
				 * WAS a[i] WILL BE NOW a[i] - each_dif
				 */
				taka += how_many * (a[i] - each_dif);
				assert (how_many < num);
				k = 0;
			}
			ans += taka;
		}
	cout << ans;
}

int32_t main() {
	ios_base::sync_with_stdio (false);
	cin.tie (0);
	solve();
	return 0;
}
