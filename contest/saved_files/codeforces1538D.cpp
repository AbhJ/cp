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
#define rep(i, a, b) for (int i = a; i <= b; i++)
#define per(i, b, a) for (int i = b; i >= a; i--)
#define all(x)       x.begin(), x.end()
using namespace std;
const int N    =     1e6 + 10;
int a, b, k;
int f(int n) {
	int cnt = 0;
	for (int i = 2; i * i <= n; i++)while (n % i == 0) {
			cnt++;
			n /= i;
		}
	return cnt + (n != 1);
}
void solve() {
	cin >> a >> b >> k;
	int mi = 0;
	if (a != b) {
		if ((a % b == 0) or (b % a == 0))
			mi = 1;
		else
			mi = 2;
	}
	if (k == 1) {
		cout << (mi == 1 ? "YES" : "NO");
		return;
	}
	int ma = f(a) + f(b);
	cout << (k >= mi and k <= ma ? "YES" : "NO");
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}