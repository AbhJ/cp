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
int n, a[N];
void solve(){
	cin >> n;
	rep(i, 1LL, n){
		cin >> a[i];
	}
	vi v;
	if(n == 1LL){
	}
	else if(n == 2LL){
		if(a[1LL] == 2LL){
			v.pb(1LL);
			swap(a[1LL], a[2LL]);
		}
	}
	else{
		rep(i, 1LL, n - 3LL){
			rep(j, i + 1, n){
				if(a[j]==i){
					if((((int)v.size() ) & 1) != (j & 1)){
						if(((int)v.size() & 1) == (n & 1)){
							v.pb(n - 1LL);
							swap(a[n - 1LL], a[n]);
						}
						else{
							v.pb(n - 2LL);
							swap(a[n - 2LL], a[n - 1LL]);
						}
					}
					per(pos, j - 1, i){
						v.pb(pos);
						swap(a[pos], a[pos + 1LL]);
					}
				}
			}
		}
	}
	while(a[n] != n or a[n - 1LL] != n - 1LL or a[n - 2LL] != n - 2LL){
		if(((int)v.size() & 1LL) == (n & 1LL)){
			v.pb(n - 1LL);
			swap(a[n - 1LL], a[n]);
		}
		else{
			v.pb(n - 2LL);
			swap(a[n - 2LL], a[n - 1LL]);
		}
	}
	assert((int)v.size() <= n * n);
	assert(is_sorted(a + 1LL, a + 1LL + n));
	cout << v.size() << "\n";
	for(auto &i: v){
		cout << i << " ";
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