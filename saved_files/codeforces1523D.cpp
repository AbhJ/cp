//Coded by Abhijay Mitra (AbJ)
#include "bits/stdc++.h"
#define double       long double
#define int          long long int
#define ibs          ios_base::sync_with_stdio(false)
#define cti          cin.tie(0)
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
const int inf  =     1e18 + 10;
const int N    =     2e6 + 10;
int n, m, p, b[N], ma[N], ans, subs[N], fina;char a[N][61];
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
void solve(){
	cin >> n >> m >> p;
	rep(i, 1LL, n){
		cin >> a[i];
		rep(j, 0, m - 1LL){
			if(a[i][j] == '1'){
				b[i] |= 1LL << j;
			}
		}
	}
	vi ind;
	rep(i, 0, 30){
		ind.pb(uniform_int_distribution<int>(1LL, n)(rng));
	}
	for(auto &i: ind){
		vi v;
		rep(j, 0, m - 1LL){
			if((b[i] >> j) & 1LL)
				v.pb(j);
		}
		vi cn(1LL << (int)v.size());
		// README
		// reduced mask array
		rep(j, 1LL, n){
			int ma = 0;
			rep(I, 0, (int)v.size() - 1LL){
				if((b[j] >> v[I]) & 1LL)
					ma |= 1LL << I;
			}
			cn[ma]++;
		}
		rep(j, 0, (int)cn.size() - 1LL){
			for (int s=j; s; s=(s-1LL)&j)if(j != s){
				cn[s] += cn[j];
			}
		}
		rep(j, 0, (int)cn.size() - 1LL){
			if(cn[j] >= (n + 1LL >> 1LL) and __builtin_popcount(j) > ans){
				ans = __builtin_popcount(j);fina = 0;
				rep(J, 0, (int)v.size() - 1LL){
					if((j >> J) & 1LL){
						fina |= 1LL << v[J];
					}
				}
			}
		}
	}
	rep(i, 0, m - 1LL){
		cout << ((fina >> i) & 1LL);
	}
}
int32_t main(){
	ibs;cti;
	solve(); return 0;
	int t; cin >> t;
	while (t--){
		solve(); cout << "\n";
	}
	return 0;
}