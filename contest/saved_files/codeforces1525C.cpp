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
#define deb(x) cout << "\n"                           \
                    << "[" << #x << " = " << x << "]" \
                    << "\n";
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
int n, m, a[N], ti[N];char s[N];
//x follows y to right
//they meet at X
//2(m - y) + (y - X) = (X - x)
//X = (x + y) / 2 + (m - y)
int f(int x, int y){
	assert(abs(x & 1) == abs(y & 1));
	return ((y - x) >> 1) + (m - y);
}
void g(deque<tuple<int, int>>d){
	while(d.size() > 1){
		auto [pos, i] = d.back();
		d.pop_back();
		auto [pos_pre, j] = d.back(); 
		d.pop_back();
		ti[i] = ti[j] = f(pos_pre, pos);
	}
}
void solve() {
	cin >> n >> m;
	rep(i, 1, n){
		cin >> a[i];
		ti[i] = - 1;
	}
	rep(i, 1, n){
		cin >> s[i];
	}
	multiset<pii> l, r;
	multiset<tuple<int, int, int>>tot;
	rep(i, 1, n){
		if(s[i] == 'L')
			l.emplace(a[i], i);
		if(s[i] == 'R')
			r.emplace(a[i], i);
	}
	// cout << l.size() << r.size() << "\n";
	for(auto [pos, i]: r)if(~ti[i] == 0){
		if(l.empty() == 0 and l.rbegin() -> F >= pos){
			auto ne = l.lower_bound(mp(pos, - inf));
			assert(ne -> F >= pos);
			// cout << ne -> S;
			if((pos + ne -> S) & 1LL)
				continue;
			ti[i] = ti[ne -> S] = ((ne -> S - pos) >> 1);
			l.erase(ne);
		}
	}
	//removed inital collision
	for(auto [pos, i]: r)if(~ti[i] == 0){
		tot.emplace(pos, 0, i);
	}
	for(auto [pos, i]: l)if(~ti[i] == 0){
		tot.emplace(pos, 1, i);
	}
	deque<tuple<int, int>>d[2];
	for(auto [pos, ty, i]: tot){
		assert(~ti[i] == 0);
		if(ty == 0){
			d[pos & 1].pb(pos, i);
		}
		if(ty == 1){
			if(d[pos & 1].empty() == 0){
				auto [pos_pre, j] = d[pos & 1].back();
				d[pos & 1].pop_back();
				ti[i] = ti[j] = ((pos - pos_pre) >> 1);
			}
			else{
				//parity remains same after inversing sign
				d[pos & 1].emplace_front(-pos, i);
			}
		}
	}
	g(d[0]);
	g(d[1]);
	rep(i, 1, n){
		cout << ti[i] << " ";
	}
}
int32_t main() {
    ibs;cti;
    //solve(); return 0;
    int xx = 0;
    int t;
    cin >> t;
    while (t--) {
        xx++;
        solve();
        cout << "\n";
    }
    return 0;
}