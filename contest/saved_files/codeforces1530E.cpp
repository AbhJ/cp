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
void ff(vi& fr, vi& ind) {
	for (int i = 0; i < (int)fr.size(); i++)
		cout << string(fr[i], char('a' + ind[i]));
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
string s; int n, f[26];
void solve() {
	cin >> s; n = s.size();
	fill(f, f + 26, 0);
	bool F = 0;
	for (int i = 0; i < n; i++)
		f[s[i] - 'a']++;
	vi fr, ind;
	for (int i = 0; i < 26; i++) if (f[i] > 0)
			fr.pb(f[i]), ind.pb(i);
	int m = fr.size();
	for (int i = 0; i < fr.size(); i++)
		if (fr[i] == 1) {
			cout << char('a' + ind[i]), fr[i] = 0, ff(fr, ind), F = 1; break;
		}
	if (F) return;
	int re = fr[0], su = 0;
	for (int i = 1; i < m; i++)
		su += fr[i];
	if (m == 1) {
		ff(fr, ind);
		return;
	}
	// debug(su);
	if (re <= su + 2LL) {
		int ch = min(2LL, fr[0]);
		for (int i = 0; i < ch; i++) cout << char('a' + ind[0]);
		fr[0] -= ch; fr[1] -= 1LL;
		cout << char('a' + ind[1]);
		for (int i = 1; i < m; i++)
			for (int j = 0; j < fr[i]; j++) {
				if (fr[0]) cout << char('a' + ind[0]), fr[0]--;
				cout << char('a' + ind[i]);
			}
		// debug(su);
		if (fr[0]) cout << char('a' + ind[0]);
		return;
	}
	// debug(su);
	if (m == 2) {
		cout << string(1, char('a' + ind[0])) + string(fr[1], char('a' + ind[1])) + string(fr[0] - 1, char('a' + ind[0]));
		return;
	}
	cout << char('a' + ind[0]); fr[0]--; cout << char('a' + ind[1]); fr[1]--;
	for (int i = 0; i < fr[0]; i++) cout << char('a' + ind[0]);
	// debug(su);
	cout << char('a' + ind[2]);
	fr[0] = 0; fr[2]--;
	ff(fr, ind);
	return;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}