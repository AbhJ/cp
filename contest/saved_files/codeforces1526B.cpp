//Coded by Abhijay Mitra (AbJ)
#include "bits/stdc++.h"
#define double long double
#define int long long int
#define ibs ios_base::sync_with_stdio(false)
#define cti cin.tie(0LL)
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
const int inf = 1e18 + 10LL;
const int N = 2e6 + 10LL;
int n;
void solve(){
	cin >> n;
	int cn = n % 11;
	n -= cn * 111;
	//even multiples generated from 11
	cout << (n >= 0LL and n % 11 == 0LL ? "YES" : "NO");
}
int32_t main(){
    ibs;cti;
    // solve(); return 0LL;
    int t; cin >> t;
    while (t--){
        solve(); cout << "\n";
    }
    return 0;
}