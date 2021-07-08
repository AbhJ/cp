//Coded by Abhijay Mitra (AbJ)
#include "bits/stdc++.h"
#define double long double
#define int long long int
#define ibs ios_base::sync_with_stdio(false)
#define cti cin.tie(0)
#define mp make_pair
#define pb emplace_back
#define F first
#define S second
using vi = std::vector<int>;
using vvi = std::vector<vi>;
using pii = std::pair<int, int>;
using vpii = std::vector<pii>;
using vvpii = std::vector<vpii>;
#define rep(i, a, b) for (int i = a; i <= b; i++)
#define per(i, b, a) for (int i = b; i >= a; i--)
#define all(x) x.begin(), x.end()
using namespace std;
const int inf = 1e18 + 10;
const int N = 2e6 + 10;
int n, l[N], r[N], x, y, dp[N][2];vi a[N];
void f(int u = 1, int p = 0){
	for(auto &i: a[u])if(i != p){
		f(i, u);
		dp[u][0] += max(dp[i][1] + abs(l[u] - r[i]), dp[i][0] + abs(l[u] - l[i]));
		dp[u][1] += max(dp[i][1] + abs(r[u] - r[i]), dp[i][0] + abs(r[u] - l[i]));
	}
}
void solve(){
	cin >> n;
	rep(i, 1, n){
		cin >> l[i] >> r[i];
	}
	rep(i, 1, n - 1){
		cin >> x >> y;
		a[x].pb(y);
		a[y].pb(x);
	}
	f();
	cout << max(dp[1][0], dp[1][1]);
	rep(i, 1, n){
		a[i].clear();
		dp[i][0] = dp[i][1] = 0;
	}
}
int32_t main(){
	ibs;cti;
	// solve(); return 0;
	int t; cin >> t;
	while (t--){
		solve(); cout << "\n";
	}
	return 0;
}