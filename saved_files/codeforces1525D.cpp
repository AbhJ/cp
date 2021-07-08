//Coded by Abhijay Mitra (AbJ)
#include "bits/stdc++.h"
#define double long double
#define int long long int
#define ll int
#define ibs ios_base::sync_with_stdio(false)
#define cti cin.tie(0)
#define bp __builtin_popcount
#define pb emplace_back
#define koto_memory(a) cout << (sizeof(a) / 1048576.0) << " MB";
#define res(v) sort(all(v)), v.erase(unique(all(v)), v.end());
#define timer cerr << "Time elapsed : " << 1.0 * clock() / CLOCKS_PER_SEC << " sec " << endl;
#define deb(x) cout << "\n"                           \
<< "[" << #x << " = " << x << "]" \
<< "\n";
using vi = std::vector<int>;
using vvi = std::vector<vi>;
using pii = std::pair<int, int>;
using vpii = std::vector<pii>;
using vvpii = std::vector<vpii>;
// mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
// int pos = uniform_int_distribution<int>(l,r)(rng);
#define mp make_pair
#define rep(i, a, b) for (int i = a; i <= b; i++)
#define per(i, b, a) for (int i = b; i >= a; i--)
#define all(x) x.begin(), x.end()
using namespace std;
const int inf = 1e18 + 10;
// const double Pi = M_PI;
// const int M = 998244353;
// const int M = 1e9+7;
#define F first
#define S second
const int N = 2e6 + 10;
int n, a[N], taka;
set<int> emp;vi fil;
void solve() {
	cin >> n;
	mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
	int tmp = uniform_int_distribution<int>(0, 10)(rng);
	rep(i, 1, n){
		cin >> a[i];
	}
	vi fi, em;
	rep(i, 1, n){
		if(a[i] == 0){
			emp.emplace(i);
			em.pb(i);
		}
		if(a[i] == 1){
			fil.pb(i);
			fi.pb(i);
		}
	}
	if(tmp == 1){
		//Start greedy method
		rep(i, 0, (int)fil.size() - 1){
			int pos = fil[i];
			int lef = inf, rig = inf;
			if(*emp.rbegin() > pos){
				auto ne = emp.upper_bound(pos);
				rig = abs(*ne - pos);
			}
			if(*emp.begin() < pos){
				auto pe = prev(emp.upper_bound(pos));
				lef = abs(*pe - pos);
			}
			assert(lef != inf or rig != inf);
			if(lef <= rig){
				taka += lef;
				emp.erase(prev(emp.upper_bound(pos)));
			}
			else{
				taka += rig;
				emp.erase(emp.upper_bound(pos));
			}
		}
	}
	else taka = inf;
	//Start dp method
	int n = (int)fi.size(), m = (int)em.size();
	vi vis(m);
	vvi dp(n, vi(m, inf));
	assert(n <= m);
	rep(i, 0, n - 1){
		int mi = inf;
		if(i == 0)
			mi = 0;
		rep(j, 0, m - 1)if(vis[j] != 1){
			dp[i][j] = min(mi + abs(fi[i] - em[j]), dp[i][j]);
			if(i != 0)mi = min(mi, dp[i - 1][j]);
			if(i == n - 1)taka = min(taka, dp[i][j]);
		}
	}
	cout << (taka != inf ? taka : 0);
}
int32_t main() {
	ibs;cti;
	solve(); return 0;
	int xx = 0;
	int t;
	cin >> t;
	while (t--) {
		xx++;
		solve();
		cout << "\n";
	}
	return 0;
}