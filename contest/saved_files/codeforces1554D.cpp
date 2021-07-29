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
int n;
void solve() {
	cin >> n;
	if (n == 1) {
		cout << "a";
	}
	else if (n & 1 ^ 1) {
		cout << string(n / 2, 'a') + 'b' + string(n / 2 - 1, 'a');
	}
	else {
		cout << string(n / 2, 'a') + 'b' + string(n / 2 - 1, 'a') + 'c';
	}
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}