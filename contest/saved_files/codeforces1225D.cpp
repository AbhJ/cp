#include "bits/stdc++.h"
#define int          long long int
#define mp           make_pair
#define pb           emplace_back
#define F            first
#define S            second
using vi       =     std::vector<int>;
using vvi      =     std::vector<vi>;
using pii      =     std::pair<int, int>;
using vpii     =     std::vector<pii>;
using vvpii    =     std::vector<vpii>;
using namespace std;
const int inf  =     1e18 + 10;
const int N    =     1e5 + 10;
int n, k, cnt[N], sp[N], a[N];
void Sieve() {
	for (int i = 2; i <= N - 1; i++) {
		if (sp[i] == 0) {
			for (int j = i; j < N; j += i) {
				sp[j] = i;
			}
		}
	}
}
map<int, int>m;
#define watch(x) cerr << (#x) << " is " << (x) << "\n";
void debug(string t) {cerr << t;}
void debug(int t) {cerr << t;}
void debug(double t) {cerr << t;}
void debug(char t) {cerr << t;}
template <class T> void debug(vector <T> v);
template <class T> void debug(vector <T> v) {cerr << "[ "; for (T i : v) {debug(i); cerr << " ";} cerr << "]";}
template <class T> void debug(set <T> v);
template <class T> void debug(set <T> v) {cerr << "[ "; for (T i : v) {debug(i); cerr << " ";} cerr << "]";}
template <class T> void debug(multiset <T> v);
template <class T> void debug(multiset <T> v) {cerr << "[ "; for (T i : v) {debug(i); cerr << " ";} cerr << "]";}
template <class T, class V> void debug(pair <T, V> p);
template <class T, class V> void debug(pair <T, V> p) {cerr << "{"; debug(p.F); cerr << ","; debug(p.S); cerr << "}";}
template <class T, class V> void debug(map <T, V> v);
template <class T, class V> void debug(map <T, V> v) {cerr << "[ "; for (auto i : v) {debug(i); cerr << " ";} cerr << "]";}
pii f(int x) {
	vi v, spf;
	while (x != 1) {
		int s = sp[x];
		while (x % s == 0)
			x /= s, cnt[s]++;
		spf.pb(s);
	}
	int val = 1, tmp = 1;
	for (auto &i : spf) {
		if (cnt[i] % k != 0) {
			v.pb(cnt[i] % k);
			val *= pow(i, v.back());
			tmp *= pow(i, k - v.back());
		}
		cnt[i] = 0;
	}
	return mp(val, tmp);
}
int cn;
void solve() {
	cin >> n >> k;
	vi v;
	for (int i = 1; i <= n; i++) {
		cin >> a[i];
		auto [x, y] = f(a[i]);
		cn += m[x];
		m[y]++;
	}
	cout << cn;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	Sieve();
	solve(); return 0;
	return 0;
}