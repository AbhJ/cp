//Coded by Abhijay Mitra (AbJ)
#include "bits/stdc++.h"
#define double long double
#define int long long int
#define ll int
#define ibs ios_base::sync_with_stdio(false)
#define cti cin.tie(0)
#define bp __builtin_popcount
#define pb emplace_back
#define res(v) sort(all(v)), v.erase(unique(all(v)), v.end());
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
const int N = 2e6 + 10;
int n, a[N], d[N][3], f[3];char c[N];
void solve() {
	cin >> n;
	rep(i, 1LL, n << 1LL){
		cin >> a[i] >> c[i];
		if(c[i] == 'R')
			f[0LL]++;
		if(c[i] == 'G')
			f[1LL]++;
		if(c[i] == 'B')
			f[2LL]++;
	}
	vvi v;
	vi tmp, V;
	rep(i, 0LL, 2LL){
		rep(j, 1LL, n << 1LL){
			if(c[j] == 'R' and i == 0LL)
				tmp.pb(a[j]);
			if(c[j] == 'G' and i == 1LL)
				tmp.pb(a[j]);
			if(c[j] == 'B' and i == 2LL)
				tmp.pb(a[j]);
		}
		sort(all(tmp));
		if(f[i] & 1LL){
			// if f[i] is odd one persom is left out
			v.pb(tmp);
		}
		else{
			V = tmp;
		}
		tmp.clear();
	}
	assert((int)v.size() & 1LL ^ 1LL);
	// totally 2LL n people
	int taka = inf;
	rep(i, 0LL, (int)v.size() - 1LL){
		rep(j, 0LL, (int)v[i].size() - 1LL){
			int cur = v[i][j];
			rep(k, 0LL, (int)v.size() - 1LL)if(k != i){
				if((*v[k].begin() )<= cur){
					taka = min(taka, cur - (*prev(upper_bound(all(v[k]), cur))));
				}
				if(v[k].back() >= cur){
					taka = min(taka, (*lower_bound(all(v[k]), cur)) - cur);
				}
			}
		}
	}
	if(v.empty() == 0LL){
		int ba = inf, da = inf;
		rep(i, 0LL, (int)V.size() - 1LL){
			int cur = V[i];
			vi &v1 = *v.begin(), &v2 = v.back();
			if((*v1.begin() )<= cur){
					ba = min(ba, cur - (*prev(upper_bound(all(v1), cur))));
			}
			if(v1.back() >= cur){
				ba = min(ba, (*lower_bound(all(v1), cur)) - cur);
			}
			if((*v2.begin() )<= cur){
					da = min(da, cur - (*prev(upper_bound(all(v2), cur))));
			}
			if(v2.back() >= cur){
				da = min(da, (*lower_bound(all(v2), cur)) - cur);
			}
		}
		taka = min(taka, ba + da);
	}
	cout << (v.empty() == 0LL ? taka : 0LL);
}
int32_t main() {
    ibs;cti;
    solve(); return 0LL;
    int t;
    cin >> t;
    while (t--) {
        solve();
        cout << "\n";
    }
    return 0LL;
}