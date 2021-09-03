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
int n; string s;
map<string, string>str;
map<string, int>tot_cnt;
int cou(const string& S, const string& sub) {
	int num = 0;
	for (size_t i = 0; (i = S.find(sub, i)) != string::npos; num++, i++);
	return num;
}
string pre(string& s) {
	return s.substr(0LL, min((int)s.length(), 3LL));
}
string suf(string& s) {
	return s.substr(max((int)s.length() - 3LL, 0LL));
}
pair<string, int> f(const string& lef, const string& rig) {
	string ret = str[lef] + str[rig];
	int tot = tot_cnt[lef] + tot_cnt[rig] + cou(suf(str[lef]) + pre(str[rig]), "haha");
	if ((int)ret.length() > 6)
		ret = pre(ret) + "blablabla" + suf(ret);
	return mp(ret, tot);
}
void solve() {
	tot_cnt.clear();
	str.clear();
	s.clear();
	int n;
	cin >> n;
	string s;
	for (int i = 0; i < n; i++) {
		cin >> s;
		string op;
		cin >> op;
		if (op == "=") {
			string lef, rig;
			cin >> lef >> rig >> rig;
			tie(str[s], tot_cnt[s]) = f(lef, rig);
			continue;
		}
		if (op == ":=") {
			string S;
			cin >> S;
			tot_cnt[s] = cou(S, "haha");
			str[s] = S;
			continue;
		}
		assert(0);
	}
	cout << tot_cnt[s];
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}