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
#define deb(x) cout<<endl<<"["<<#x<<" = "<<x<<"]"<<endl;
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
const int N = 2e5 + 10;
int n, x[N], y[N], ans, tmp, cnt, sm, la, hx[N], hy[N];
void solve() {
	cin >> n;
	rep(i, 1, n){
		cin >> x[i] >> y[i];
		hx[i] = x[i];
		hy[i] = y[i];
	}
	sort(x + 1, x + 1 + n);
	sort(y + 1, y + 1 + n);
	if(x[n] - x[1] > y[n] - y[1]){
		swap(x, y);
		swap(hx, hy);
	}
	rep(i, 1, n){
		if(hy[i] == y[n])
			la = i;
		if(hy[i] == y[1])
			sm = i;
	}
	if(abs(hx[la] - hx[sm]) != x[n] - x[1])
		ans = x[n] - x[1];
	cout << max({y[n] - y[2], ans, y[n - 1] - y[1], x[n] - x[2], x[n - 1] - x[1]});
}
int32_t main() {
    ibs;cti;
    solve(); return 0;
    int t;
    cin >> t;
    while (t--) {
        solve();
        cout << "\n";
    }
    return 0;
}