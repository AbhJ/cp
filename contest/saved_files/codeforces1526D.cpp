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
string s, t, mov;const string tmp = "ANOT";int n, pos[4][N], ty[N], taka, ans;
void solve() {
	cin >> s;
	n = s.length();
	mov = tmp;
	vi freq(4);
	rep(i, 0, n - 1){
		rep(j, 0, 3){
			if(s[i] == tmp[j]){
				ty[i] = j;
				freq[j]++;
			}
		}
	}
	per(i, n - 1, 0){
		rep(j, 0, 3){
			pos[j][i] = pos[j][i + 1] + (s[i] == tmp[j]);
		}
	}	
	ans = 0;
	do{
		rep(i, 0, n - 1){
			rep(j, 0, 3){
				if(s[i] != mov[j]){
					rep(k, 0, 3){
						if(tmp[k] == mov[j])
							taka += pos[k][i];
					}
				}
				else break;
			}
		}
		if(taka >= ans){
			ans = taka;
			taka = 0;
			t.clear();
			rep(i, 0, 3){
				rep(k, 0, 3){
					if(tmp[k] == mov[i])
						t += string(freq[k], tmp[k]);
				}
			}
		}
		taka = 0;
	}while(next_permutation(all(mov)));
	cout << t;
	ans = 0;
	rep(i, 0, 3){
		rep(j, 0, n - 1)
			pos[i][j] = 0;
	}
	rep(i, 0, n - 1)
		ty[i] = 0;
	ans = 0;
}
int32_t main() {
	ibs;cti;
    //solve(); return 0;
	int t;
	cin >> t;
	while (t--) {
		solve();
		cout << "\n";
	}
	return 0;
}