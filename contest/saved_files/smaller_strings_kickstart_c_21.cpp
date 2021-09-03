#include "bits/stdc++.h"
#define double long double
#define int long long int
#define ll int
#define ibs ios_base::sync_with_stdio(false)
#define cti cin.tie(0)
#define bp __builtin_popcount
#define pb emplace_back
#define koto_memory(a) cout<<(sizeof(a)/1048576.0)<<" MB";
#define res(v) sort(all(v)),v.erase(unique(all(v)), v.end());
#define timer cerr << "Time elapsed : " << 1.0 * clock() / CLOCKS_PER_SEC << " sec "<<endl;
#define deb(x) cout<<endl<<"["<<#x<<" = "<<x<<"]"<<endl;
using vi = std::vector<int>;
using vvi = std::vector<vi>;
using pii = std::pair<int,int>;
using vpii = std::vector<pii>;
using vvpii = std::vector<vpii>;
#define mp         make_pair
#define rep(i,a,b) for (int i = a; i <= b; i++)
#define per(i,b,a) for (int i = b; i >= a; i--)
#define all(x) x.begin(), x.end()
using namespace std;
const int N=2e6+10;
const int inf = /*0x3f3f3f3f*/1e18+10;
// const ll M = 998244353 ; // Mulo
const int M = 1e9+7 ; // Mulo
// const double Pi = M_PI;
#define F first
#define S second
int n, a[N], b[N], cnt, k, ans, tmp, dp[N][4];string s;
int F(int i, int ok) {
  int tmp = (n >> 1);
  if(n & 1)tmp = (n >> 1) + 1;
  if(i == tmp)
    return 1;
  if(~dp[i][ok])
    ;
  else{
    int ans = 0;
    int cur_con = s[i] - 'a';
    if(ok == 0)
      ans += k * F(i + 1, 0);
    else
      ans += cur_con * F(i + 1, 0) + F(i + 1, 1);
    dp[i][ok] = ans % M;
    ans = dp[i][ok]; 
  }
  return dp[i][ok];
}
void solve(){
  cin >> n >> k >> s;
  memset(dp, -1, sizeof dp);
  string ba_hat, da_hat;
  int  ans = F(0, 2);
  ba_hat = s.substr(0, n >> 1);
  da_hat = ba_hat;
  reverse(da_hat.begin(), da_hat.end());
  if(n & 1)ba_hat=ba_hat+s[n >> 1];
  if((ba_hat + da_hat) >= s)
    ans--,
  ans = (ans + M) % M;
  cout << ans;
}
int32_t main()
{
  ibs;cti;
//   solve();return 0;
  int xx=0;
  int t;cin>>t;while(t--){xx++;cout<<"Case #"<<xx<<": ";solve();cout<<"\n";}
  return 0;
}