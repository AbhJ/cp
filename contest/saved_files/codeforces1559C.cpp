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
int a[N];
void solve() {
	int n;
	cin >> n;
	for (int i = 1; i <= n; i++) {
		cin >> a[i - 1];
	}
	if (a[0] != 0) {
		cout << n + 1 << " ";
		for (int i = 1; i <= n; i++) cout << i << " ";
	}
	else if (a[n - 1] == 0)
		for (int i = 1; i <= n + 1; i++) cout << i << " ";
	else {
		for (int i = 0; i < n - 1; i++) {
			if (a[i] == 0 and a[i + 1] != 0) {
				for (int j = 1; j <= i + 1; j++)
					cout << j << " ";
				cout << n + 1 << " ";
				for (int j = i + 2; j <= n; j++)
					cout << j << " ";
				return;
			}
		}
		cout << - 1;
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