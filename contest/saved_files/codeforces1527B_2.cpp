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
string s;int n, cnt_zer;
bool is_palindrome(const string &s) {
	return equal(s.begin(), s.begin() + s.size() / 2, s.rbegin());
}
void solve() {
	cin >> n >> s;
	cnt_zer = count(all(s), '0');
	if(is_palindrome(s)){
		if(cnt_zer >= 3 and cnt_zer & 1){
			cout << "ALICE";
		}
		else{
			cout << "BOB";
		}
	}
	else{
		if(cnt_zer == 0)
			cout << "DRAW";
		else if(n & 1 and cnt_zer == 2 and s[n >> 1] == '0'){
			//ALICE reverses
			//BOB fills non middle
			//ALICE fills middle (no reverse as palindrome)
			cout << "DRAW";
		}
		else
			cout << "ALICE";
		//ALICE reverse
		//BOB fills
	}
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