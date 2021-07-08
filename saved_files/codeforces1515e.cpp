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
const int N = 4e2 + 10;
int n, M, dp[N][N], f[N], b[N];
int powM(int b, int p) {
	int r=1;
	for(; p; b=b*b%M, p/=2)
		if(p&1)
			r=r*b%M;
		return r;
}
int mul(int a,int b){
	return ((a)*(b))%M;
}
int add(int a,int b){
	a+=b;
	if(a>=M)a-=M;
	return a;
}
int sub(int a,int b){
	return ((a%M)-(b%M)+M)%M;
}
void calc_fact(){
	f[0]=b[0]=1;
	rep(i,1,N-1){
		f[i]=mul(f[i-1],i);
		b[i]=powM(f[i],M-2);
	}
}
int C(int n,int r){
	assert(n<N);assert(r<N);
	if(r>n)return 0;
	return (f[n]*mul(b[r],b[n-r])%M)%M;
}
//i is total length of computers on and j is length of manual on
void solve(){
	cin >> n >> M;
	calc_fact();
	dp[1][1] = 1;
	// i - j - k computers were set on automatically, k computers were already set manually j - 1 are being set by manual increasing total number of set computers to i - 1
	rep(i, 2, n){
		dp[i][i] = powM(2, i - 1);
		rep(j, 2, i - 1){
			rep(k, 0, i - j){
				dp[i][j + k - 1] = add(dp[i][j + k - 1], mul(dp[i - j][k], mul(C(j + k - 1, k), powM(2, j - 2))));
			}
		}
	}
	int su = 0;
	rep(i, 0, n){
		su = add(su, dp[n][i]);
	}
	cout << su;
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