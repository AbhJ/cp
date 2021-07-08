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
int k, n;string s;
void solve(){
	cin >> n >> k;
	string s;
	per(i, k, 1){
		s += (char)(i - 1 + 'a');
		per(j, i - 1, 1){
			s += string(1, (char)(i - 1 + 'a')) + (char)(j - 1 + 'a');
		}
	}
	rep(i, 0, n - 1){
		cout << s[i % (int)(s.length())];
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