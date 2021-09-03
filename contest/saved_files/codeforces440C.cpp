
/**
 * @author      : abhj
 * @created     : Tuesday Aug 03, 2021 12:33:20 IST
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
int n, ans, a[N], siz, tmp;
string s;
map<pii, int>dp;
// ITS BETTER TO ALWAYS ADD UNLESS WE
// ARE FORCED TO SUBRACT.

int f(int num, int siz) {
	int one_string = a[siz];
	int cnt_one_string = num / one_string;
	int remain = num % one_string;
	if(dp.count(mp(num, siz)))
		return dp[mp(num, siz)];
	if(remain == 0)
		return cnt_one_string * siz;
	int next_string_add = remain;
	int next_string_sub = one_string - remain;

	// WE CAN SUBTRACT next_string_add FROM one_string TO
	// REPRESENT THE CASE OF SUBTRACTION. SINCE IT HAS
	// siz ONES, WE NEED TO PAY EXTRA siz ONES TO ENTER
	// THAT CASE. WE SHALL RETURN THE MINIMUM COST OF BOTH

	return dp[mp(num, siz)] = cnt_one_string * siz + min(f(next_string_add, siz - 1),
							  f(next_string_sub, siz - 1) + siz);
}
void solve() {
	cin >> n;
	for(int i = 1; i < 17; i++)
		a[i] = a[i - 1] * 10 + 1;
	s = to_string(n);
	siz = (int)s.length();
	cout << f(n, siz + 1);
}

int32_t main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	solve();
	return 0;
}
