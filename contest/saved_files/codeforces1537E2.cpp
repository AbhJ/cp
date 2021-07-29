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
int n, k; string s, t;
void solve() {
	cin >> n >> k >> s;
	int repeaterIndex = 1;
	for (int i = 1; i < n; i++) {
		if (s[i] > s[i % repeaterIndex]) {
			// current position is better
			// copying would make worse
			break;
		}
		if (s[i] < s[i % repeaterIndex])
			repeaterIndex = i + 1;
	}
	for (int i = 0; i < k; i++) {
		cout << s[i % repeaterIndex];
	}
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	solve(); return 0;
	return 0;
}