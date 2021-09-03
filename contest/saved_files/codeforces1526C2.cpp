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
const int N = 2e6 + 10;
int n, a[N], ans, dp[N];priority_queue<int>q;
void solve() {
  cin >> n;
  rep(i, 1, n){
    cin >> a[i];
    a[i] = - a[i];
  }
  int su = 0, cnt = 0;
  q.emplace(0);
  rep(i, 1, n){
    su += a[i];
    q.emplace(a[i]);
    while(su > 0 and q.empty() == 0){
      su -= q.top();
      q.pop();
      cnt--;
    }
  }
  cout << n + cnt;
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