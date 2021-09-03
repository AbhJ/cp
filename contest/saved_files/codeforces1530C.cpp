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
const int N    =     2e6 + 10;
int n, bam, dan, a[N], b[N], koita; multiset<int> remo, keep, remo2, keep2;
void f(multiset<int>& remo, multiset<int>& keep, int a[N], int &sum) {
	for (int i = 1; i <= n; i++) {
		remo.insert(a[i]);
		if (remo.size() > n / 4) {
			sum += *remo.rbegin();
			keep.insert(*remo.rbegin());
			remo.erase(remo.find(*remo.rbegin()));
		}
	}
}
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
void solve() {
	cin >> n;
	for (int i = 1; i <= n; i++) {
		cin >> a[i];
	}
	for (int i = 1; i <= n; i++) {
		cin >> b[i];
	}
	remo.clear(); keep.clear(); remo2.clear(); keep2.clear(); bam = dan = 0;
	koita = 0;
	f(remo, keep, a, bam);
	f(remo2, keep2, b, dan);
	while (bam < dan) {
		koita++;
		int newn = n + koita;
		bam += 100;
		keep.insert(100);
		if (keep.size() > newn - newn / 4) bam -= *keep.begin(), keep.erase(keep.begin());
		remo2.insert(0);
		if (remo2.size() > newn / 4) dan += *remo2.rbegin(), remo2.erase(remo2.find(*remo2.rbegin()));
	}
	cout << koita;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}