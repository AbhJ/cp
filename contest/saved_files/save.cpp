#include "bits/stdc++.h"
#define double long double
#define int long long int
#define per(i,b,a) for (int i = b; i >= a; i--)
#define pb emplace_back
#define res(v) sort(all(v)),v.erase(unique(all(v)), v.end());
using vi = std::vector<int>;
using vvi = std::vector<vi>;
#define all(x)     x.begin(), x.end()
using pii = std::pair<int, int>;
#define S second
using vpii = std::vector<pii>;
using vvpii = std::vector<vpii>;
#define F first
#define mp make_pair
#define rep(i,a,b) for (int i = a; i <= b; i++)
using namespace std;
#define ll int
int M = 1e9 + 7;
int gcdExtended(int a, int b, int *x, int *y);

// Function to find modulo inverse of b. It returns
// -1 when inverse doesn't
int modInverse(int b, int m)
{
	int x, y; // used in extended GCD algorithm
	int g = gcdExtended(b, m, &x, &y);

// Return -1 if b and m are not co-prime
	if (g != 1)
		return -1;

// m is added to handle negative x
	return (x % m + m) % m;
}
int div(int a, int b, int m = M)
{
	a = a % m;
	int inv = modInverse(b, m);
// if (inv == -1)
//    cout << "Division not defined";
// else
	return (inv * a) % m;
}

// C function for extended Euclidean Algorithm (used to
// find modular inverse.
int gcdExtended(int a, int b, int *x, int *y)
{
// Base Case
	if (a == 0)
	{
		*x = 0, *y = 1;
		return b;
	}

	int x1, y1; // To store results of recursive call
	int gcd = gcdExtended(b % a, a, &x1, &y1);

// Update x and y using results of recursive
// call
	*x = y1 - (b / a) * x1;
	*y = x1;

	return gcd;
}
const int inf = 1e18 + 10;
// const ll M = 998244353;
// const int M = 1e9+7;
const int N = 4e6 + 10;
int n, m;
void solve() {
	cin >> n >> m;
	m++;
	int ans = 0;
	for (int i = log2(max(n, m)); i >= 0; i--) {
		int dan = (m >> i) & 1LL,
		    bam = (n >> i) & 1LL;
		if (bam == 1 and dan == 0)
			// DONE
			break;
		if (bam == 0 and dan == 1)
			ans |= (1LL << i);
	}
	cout << ans;
}
int32_t main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	int t = 1;
	cin >> t;
	while (t--)solve(), cout << "\n";
	return 0;
}