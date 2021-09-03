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
int a[N], n; set<int> used;
void f() {
	cout << n - 1 << "\n";
	int remo;
	for (int i = 0; i < n; i++) {
		if (used.find(i) == used.end()) {
			remo = i;
			break;
		}
	}
	set<int> done; vi ans(n, - 1);
	ans[remo] = a[remo];
	done.insert(a[remo]);
	for (int i = 0; i < n; i++) if (i != remo)
			if (!done.count(a[i])) done.insert(a[i]), ans[i] = a[i];
			else done.insert(remo), ans[i] = remo;
	for (int i = 0; i < n; i++) cout << ++ans[i] << " ";
}
void g() {
	cout << used.size() << "\n"; used.clear();
	vi v;
	vi hatao;
	for (int i = 0; i < n; i++)
		if (!used.count(a[i])) used.insert(a[i]);
		else v.pb(i);
	for (int i = 0; i < n; i++)
		if (!used.count(i))hatao.pb(i);
	sort(v.begin(), v.end());
	if (hatao == v)
		for (int i = 0; i < hatao.size(); i++)
			a[v[i]] = hatao[(i + 1) % hatao.size()];
	else {
		set<int> s;
		for (auto &x : hatao)s.insert(x);
		for (auto &i : v) {
			if (s.find(i) != s.end()) {
				if (*s.begin() != i)
					a[i] = *s.begin(), s.erase(s.begin());
				else
					a[i] = *s.rbegin(), s.erase(s.find(*s.rbegin()));
				i = -1;
			}
		}
		for (int i = 0; i < v.size(); i++)
			if (v[i] != -1) a[v[i]] = *s.begin(), s.erase(s.begin());
	}
	for (int i = 0; i < n; i++)cout << ++a[i] << " ";
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
	used.clear();
	for (int i = 0; i < n; i++) {
		cin >> a[i];
		a[i]--;
		used.insert(a[i]);
	}
	if (used.size() == n) {
		cout << n << "\n";
		for (int i = 0; i < n; i++)cout << a[i] + 1 << " ";
		return;
	}
	if (used.size() == n - 1) f();
	else g();
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}