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
int a, b, c;
// SOLVING LINEAR DIOPHANTINE EQUATION
// https://math.libretexts.org/Courses/Mount_Royal_University/MATH_2150%3A_Higher_Arithmetic/5%3A_Diophantine_Equations/5.1%3A_Linear_Diophantine_Equations
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
void solve() {
	cin >> a >> b >> c;
	int x, y, g = gcdExtended(a, b, &x, &y);
	if (c % g != 0) {
		cout << - 1;
		return;
	}
	x = (c / g) * x;
	y = (c / g) * y;
	cout << - x << " " << - y;
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	solve(); return 0;
	return 0;
}