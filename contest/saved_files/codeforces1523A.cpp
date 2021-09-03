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
int n, m;
string s;
void solve(){
	cin >> n >> m >> s;
	vi v;
	v.pb(- inf);
	rep(i, 0, n - 1){
		if(s[i] == '1')
			v.pb(i);
	}
	v.pb(inf);
	rep(i, 0, n - 1){
		if(*lower_bound(all(v), i) - i == i - *prev(upper_bound(all(v), i)) and *lower_bound(all(v), i) - i <= m and i - *prev(upper_bound(all(v), i)) and s[i] == '0')
			cout << 0;
		else if(*lower_bound(all(v), i) - i <= m)
			cout << 1;
		else if(i - *prev(upper_bound(all(v), i)) <= m)
			cout << 1;
		else
			cout << 0;
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