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
const int N    =     2e7 + 10;
int c, d, x, a, b;
int sp[N], koi[N];
void Sieve(){
	rep(i, 2, N - 1){
		if(sp[i] == 0){
			for(int j=i;j<N;j+=i){
				sp[j] = i;
			}
		}
	}
	rep(i, 2, N - 1){
		koi[i] = koi[i / sp[i]] + (sp[i / sp[i]] != sp[i]);
		// koi is number of prime factors of i
	}
}
void solve(){
	cin >> c >> d >> x;
	int cnt = 0;
	// README
	// cL - dG = x
	// (cy - d)G = x
	// y == (x / g + d) / c
	// all prime divisors of y is wanted
	rep(g, 1, sqrt(x))if(x % g == 0){
		(x / g + d) % c == 0 ? cnt += 1 << koi[(x / g + d) / c] : 1;
		if(x != g * g)
			(g + d) % c == 0 ? cnt += 1 << koi[(g + d) / c] : 1;
	}
	cout << cnt;
}
int32_t main(){
	ibs;cti;
	Sieve();
	// solve(); return 0;
	int t; cin >> t;
	while (t--){
		solve(); cout << "\n";
	}
	return 0;
}