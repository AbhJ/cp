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
const int N = 5e3 + 10;
int n, a[N][N], co[N], m;vi A[N];
int dfs(vi nod){
    int ma = 0, pos = 0, rooot;
    int n = (int)nod.size();
    rep(i, 0, n - 1){
        rep(j, i + 1, n - 1){
            ma = max(ma, a[nod[i]][nod[j]]);
        }
    }
    if(n == 1){
        co[nod.back()] = a[nod.back()][nod.back()];
        return nod.back();
    }
    // vector<vector<pii>>
    rooot = ++m;
    vi vis(n);
    rep(i, 0, n - 1)if(vis[i] == 0){
        vi subt{nod[i]};
        vis[i] = 1;
        rep(j, 0, n - 1)if(vis[j] == 0){
            if(ma > a[nod[i]][nod[j]]){
                vis[j] = 1;
                subt.pb(nod[j]);
            }
        }
        A[rooot].pb(dfs(subt));
    }
    co[rooot] = ma;
    return rooot;
}
void solve() {
    cin >> n;
    rep(i, 1, n){
        rep(j, 1, n){
            cin >> a[i][j];
            if(i == j)
                co[i] = a[i][j];
        }
    }
    m = n;
    //m is total number of nodes
    vi v(m);
    iota(all(v), 1);
    int baaap = dfs(v);
    cout << m;
    cout << "\n";
    rep(i, 1, m){
        cout << co[i] << " ";
    }
    cout << "\n";
    cout << baaap << "\n";
    rep(i, 1, m){
        for(auto &j: A[i]){
            cout << j << " " << i << "\n";
        }
    }
}
int32_t main() {
    ibs;cti;
    solve(); return 0;
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