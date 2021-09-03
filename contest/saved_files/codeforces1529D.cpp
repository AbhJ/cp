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
#define rep(i, a, b, j) for (int i = a; i <= b; i += j)
#define per(i, b, a) for (int i = b; i >= a; i--)
#define all(x) x.begin(), x.end()
using namespace std;
const int inf = 1e18 + 10;
const int N = 1e6 + 10;
const int M = 998244353;
int n, dp[N];
void solve(){
	cin >> n;
    rep(i, 1, n, 1){
    	rep(j, i, n, i){
    		dp[j]++;
    	}
    }
    int runn = 0;
    rep(i, 1, n, 1){
    	dp[i] = (dp[i] + runn) % M;
    	runn += dp[i];
    	runn %= M;
    }
	cout << dp[n];
}
int32_t main(){
    ibs;cti;
    // https://www.youtube.com/watch?v=hZAJ-VLYflw describes the diagram awesomely
    solve(); return 0;
    int t; cin >> t;
    while (t--){
        solve(); cout << "\n";
    }
    return 0;
}