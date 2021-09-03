//Coded by Abhijay Mitra
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <cstring>
#include <numeric>
#include <vector>
#include <iterator> 
#include <map>
#include <set>
#include <climits>
#include <queue>
#include <cmath>
#include <stack>
#include <cctype>
#include <bitset>
// #include <bits/stdc++.h>
#define double long double
#define int long long int
#define ll int
#define ibs ios_base::sync_with_stdio(false)
#define cti cin.tie(0)
#define bp __builtin_popcount
#define pb push_back
#define koto_memory(a) cout<<(sizeof(a)/1048576.0)<<" MB";
#define res(v) v.resize(unique(v.begin(), v.end()) - v.begin());
#define cnt_res(v) std::distance(v.begin(),std::unique(v.begin(), v.end())); 
#define timer cerr << "Time elapsed : " << 1.0 * clock() / CLOCKS_PER_SEC << " sec \n";
using vi = std::vector<int>;
using vvi = std::vector<vi>;
using pii = std::pair<int,int>;
using vpii = std::vector<pii>;
using vvpii = std::vector<vpii>;
#define mp         make_pair
#define rep(i,a,b) for (int i = a; i <= b; i++)
#define per(i,b,a) for (int i = b; i >= a; i--)
#define all(x)     x.begin(), x.end()
using namespace std;
const int N=1e5+10;
const int inf = /*0x3f3f3f3f*/1e18+10;
// const ll M = 998244353 ; // Mulo
// const int M = 1e9+7 ; // Mulo
const double Pi = 3.14159265;
#define F first
#define S second
int n,m,z,l,r,t[4*N][30],x[4*N],a[N];
//x contains the xor
//t[i][j] contains num for jth bit
//building the tree
void build(int a[], int v=1, int tl=1, int tr=n) {
    if (tl == tr) {
    	rep(i,0,29)
        	if((1<<i)&a[tl])
        		t[v][i]=1;
    } else {
        int tm = (tl + tr) / 2;
        build(a, v*2, tl, tm);
        build(a, v*2+1, tm+1, tr);
       rep(i,0,29)
        	t[v][i]=t[2*v][i]+t[2*v+1][i];
    }
}
void push(int v,int tl,int tr){
	int tm=tl+tr>>1;
	if(x[v]==0)return;
	rep(i,0,29){
		if((1<<i)&x[v]){
			t[2*v][i]=tm-tl+1-t[2*v][i];
			t[2*v+1][i]=tr-(tm+1)+1-t[2*v+1][i];
		}
	}
	x[v*2]^=x[v];
	x[v*2+1]^=x[v];
	x[v]=0;
}
//queries
int sum(int l, int r,int v=1, int tl=1, int tr=n ) {
    if (l>r or tl>r or tr<l)return 0;
    int cnt=0;
    if (tl>=l and tr<=r){
    	rep(i,0,29){
    		cnt+=(1<<i)*t[v][i];
    	}
    	return cnt;
	}
	push(v,tl,tr);
    int tm=(tl+tr)/2;
    return sum( l,r,v*2, tl, tm)
           + sum(l, r,v*2+1, tm+1, tr );
}
void update(int l,int r,int val,int tl=1,int tr=n,int v=1){
  if(tl>tr or tl>r or tr<l)return;
  if(tl>=l and tr<=r){
  	x[v]^=val;
  	rep(i,0,29)
  		if((1<<i)&val){
  			t[v][i]=tr-tl+1-t[v][i];
  		}
  	return;
  }
  push(v,tl,tr);
  int mid=(tr+tl)/2;
  update(l,r,val,tl,mid,v*2);
  update(l,r,val,mid+1,tr,v*2+1);
  rep(i,0,29)t[v][i]=t[v*2][i]+t[v*2+1][i];
}
void solve(){
	cin>>n;
	rep(i,1,n){cin>>a[i];}
	build(a);
	cin>>m;
	rep(i,1,m){
		cin>>z>>l>>r;
		if(z==1){
			cout<<sum(l,r)<<"\n";
		}
		if(z==2){
			int va;cin>>va;
			update(l,r,va);
		}
	}
}
int32_t main()
{
  ibs;cti;
  solve()
  /*,cout<<"\n"*/;
  // cout<<"\n";
  int xx=0;
  // int t;cin>>t;while(t--){/*xx++;cout<<"Case "<<xx<<":\n"*/;solve();cout<<"\n";}
  return 0;
}