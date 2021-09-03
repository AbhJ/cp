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
int n, k;pii a[N];
void solve(){
	cin >> n >> k;
	rep(i, 1, n){
		cin >> a[i].F >> a[i].S;
	}
	sort(a + 1, a + 1 + n);
	rep(i, 1, n){
		if(a[i].F - a[i - 1].F <= k){
			k += a[i].S - a[i].F + a[i - 1].F;
		}
		else{
			cout << a[i - 1].F + k;
			return;
		}
	}
	cout << a[n].F + k;
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