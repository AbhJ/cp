

/**
 * @author      : abhj
 * @created     : Sunday Aug 22, 2021 09:40:07 IST
 * @filename    : a.cpp
 */

#include "bits/stdc++.h"
#define int          long long int
#define mp           make_pair
#define pb           emplace_back
#define F            first
#define S            second
using vi       =     std::vector<int>;
using vvi      =     std::vector<vi>;
using pii      =     std::pair<int, char>;
using vpii     =     std::vector<pii>;
using vvpii    =     std::vector<vpii>;
using namespace std;
const int inf  =     1e18 + 10;
const int N    =     2e6 + 10;

int n, maxa, cnt;
string s, S;

static bool cmp (pii a, pii b) {
	if (a.F > b.F)
		return 1;
	if (a.F == b.F)
		return a.S > b.S;
	return 0;
}
#define watch(x) cerr << (#x) << " is " << (x) << "\n";
void debug(string t) {
	cerr << t;
}
void debug(int t) {
	cerr << t;
}
void debug(double t) {
	cerr << t;
}
void debug(char t) {
	cerr << t;
}
template <class T> void debug(vector <T> v);
template <class T> void debug(vector <T> v) {
	cerr << "[ ";
	for (T i : v) {
		debug(i);
		cerr << " ";
	}
	cerr << "]";
}
template <class T> void debug(set <T> v);
template <class T> void debug(set <T> v) {
	cerr << "[ ";
	for (T i : v) {
		debug(i);
		cerr << " ";
	}
	cerr << "]";
}
template <class T> void debug(multiset <T> v);
template <class T> void debug(multiset <T> v) {
	cerr << "[ ";
	for (T i : v) {
		debug(i);
		cerr << " ";
	}
	cerr << "]";
}
template <class T, class V> void debug(pair <T, V> konta);
template <class T, class V> void debug(pair <T, V> konta) {
	cerr << "{";
	debug(konta.F);
	cerr << ",";
	debug(konta.S);
	cerr << "}";
}
template <class T, class V> void debug(map <T, V> v);
template <class T, class V> void debug(map <T, V> v) {
	cerr << "[ ";
	for (auto i : v) {
		debug(i);
		cerr << " ";
	}
	cerr << "]";
}

vpii v;

void solve () {
	cin >> s;
	n = s.length();

	maxa = 0;

	map<char, int> f;

	for (auto charac : s)
		f[charac]++,
		maxa = max(maxa, f[charac]);

	if (maxa > n / 2) {
		cout << "IMPOSSIBLE";
		return;
	}

	v.clear();

	for (auto konta : f)v.pb(konta.S, konta.F);

	sort(v.begin(), v.end(), cmp);

	S = s;

	cnt = 0;

	for (int i = 0; i < v.size(); i++)
		for (int j = 0; j < v[i].F; j++)
			S[cnt++] = v[i].S;

	string T = S.substr(maxa) + S.substr(0, maxa);

// debug(T)

	map<char, set<int>> ma;

	for (int i = 0; i < n; i++)
		ma[s[i]].insert(i);

	string ans = s;

	for (int i = 0; i < n; i++)
		ans[*ma[S[i]].begin()] = T[i], ma[S[i]].erase(ma[S[i]].begin());

	cout << ans;

}
int32_t main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	int t, x = 0;
	cin >> t;
	while (t--) {
		cout << "Case #" << ++x << ": ";
		solve();
		cout << "\n";
	}
	return 0;
}
