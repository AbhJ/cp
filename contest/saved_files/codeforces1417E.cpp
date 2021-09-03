#include "bits/stdc++.h"
#define mp           make_pair
#define pb           emplace_back
#define F            first
#define S            second
using vi       =     std::vector<int>;
using vvi      =     std::vector<vi>;
using pii      =     std::pair<int, int>;
using vpii     =     std::vector<pii>;
using vvpii    =     std::vector<vpii>;
#define rep(i, a, b) for (int i = a; i <= b; i++)
#define per(i, b, a) for (int i = b; i >= a; i--)
#define all(x)       x.begin(), x.end()
using namespace std;
const int inf  =     1e18 + 10;
const int N    =     3e5 + 10;
int n, a[N], c[32]; long long x, cn;
int getInvCount(int arr[], int n)
{
	// Create an empty set and insert first element in it
	multiset<int> set1;
	set1.insert(arr[0]);

	int invcount = 0; // Initialize result

	multiset<int>::iterator itset1; // Iterator for the set

	// Traverse all elements starting from second
	for (int i = 1; i < n; i++)
	{
		// Insert arr[i] in set (Note that set maintains
		// sorted order)
		set1.insert(arr[i]);

		// Set the iterator to first greater element than arr[i]
		// in set (Note that set stores arr[0],.., arr[i-1]
		itset1 = set1.upper_bound(arr[i]);

		// Get distance of first greater element from end
		// and this distance is count of greater elements
		// on left side of arr[i]
		invcount += distance(itset1, set1.end());
	}

	return invcount;
}
unordered_map<long long, int>cnt;
void f(int bit) {
	cnt.clear();
	long long inv = 0, inv2 = 0;
	// 1 0 inversion
	rep(i, 1, n) {
		if ((a[i] >> bit) & 1LL) {
			inv += cnt[(a[i] >> bit) ^ 1];
			// all numbers it has inversion at this stage of bit
		}
		else {
			inv2 += cnt[(a[i] >> bit) ^ 1];
			// all numbers it has inversion at this stage of bit
		}
		cnt[a[i] >> bit]++;
	}
	if (inv >= inv2) {
		cn += inv2;
	}
	else {
		x |= (1LL << bit);
		// cout << bit;
		cn += inv;
	}
	// README
	// this breaks current array into two and modifies x when necessary
}
void solve() {
	cin >> n;
	int koi = - 1;
	rep(i, 1, n) {
		cin >> a[i];
	}
	rep(bit, 0, 30) {
		f(bit);
	}
	cout << cn << " " << x;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	solve(); return 0;
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}