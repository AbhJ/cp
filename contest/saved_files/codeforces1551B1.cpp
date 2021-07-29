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
string s; vi f;
void solve() {
	cin >> s;
	f = vi (26);
	for (auto &i : s)
		f[i - 'a']++;
	int bam = 0, dan = 0;
	for (int i = 0; i < 26; i++)
		if (f[i] >= 2)
			bam++,
			    dan++;
		else if (f[i] == 1)
			if (bam < dan)
				bam++;
			else
				dan++;
	cout << min(bam, dan);
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}