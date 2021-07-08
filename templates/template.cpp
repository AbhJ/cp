#include <bits/stdc++.h>
#include <numeric>
#define int long long int
#define ibs ios_base::sync_with_stdio(false)
#define cti cin.tie(0)
#define bp __builtin_popcount
#define pb pb
#define eb emplace_back
using namespace std;//coded by abhijay mitra
const int N = 3e5 + 3;
// const int N=300;
const int M = 998244353; // modulo
#define F first
#define S second
#define MAX 100
#define MAX_CHAR 26
#define LIM 1LL<<62-1;
using namespace std;

#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
template<typename T>
using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
int pos = uniform_int_distribution<int>(l, r)(rng);
int dx[4] = {1, 0, -1, 0}, dy[4] = {0, 1, 0, -1};
int dx[] = { -2, -1, 1, 2, -2, -1, 1, 2 };
int dy[] = { -1, -2, -2, -1, 1, 2, 2, 1 };
int dfs(int u = 1, int p = 0, int level = 0) {
	int subtree = 0;
	for (auto i : a[u])if (i != p) {
			subtree += dfs(i, u, level + 1);
		}
	q.push(level - subtree);
	return subtree + 1;
}
map<int, string>m = {{1, "I"}, {4, "IV"}, {5, "V"}, {9, "IX"}, {10, "X"}, {40, "XL"}, {50, "L"},
	{90, "XC"}, {100, "C"}, {400, "CD"}, {500, "D"}, {900, "CM"}, {1000, "M"}
};
//int to roman
map<string, int>m = {{"I", 1}, {"IV", 4}, {"V", 5}, {"IX", 9}, {"X", 10}, {"XL", 40}, {"L", 50},
	{"XC", 90}, {"C", 100}, {"CD", 400}, {"D", 500}, {"CM", 900}, {"M", 1000}
};
//roman to int
int powM(int b, int p) {
	int r = 1;
	for (; p; b = b * b % M, p /= 2) {
		if (p & 1)
			r = r * b % M;
	}
	return r;
}
int mul(int a, int b) {
	return ((a) * (b)) % M;
}
int add(int a, int b) {
	a += b;
	if (a >= M)a -= M;
	return a;
}
int sub(int a, int b) {
	return ((a % M) - (b % M) + M) % M;
}
int dfs(int x, int i, int pls = 1, bool st = 0) {
	u[x] = 1;
	if (st)c[x] = i + 1;
	for (int y : e[x])if (!u[y])
			return v[i][x] + dfs(y, (i + pls) % 3, pls, st);
	return v[i][x];
}
bool is_palindrome(const string &s) {
	return equal(s.begin(), s.begin() + s.size() / 2, s.rbegin());
}
//	topological sort
int n; // number of vertices
vvi adj; // adjacency list of graph
vector<bool> visited;
vi ans;

void dfs(int v) {
	visited[v] = true;
	for (int u : adj[v]) {
		if (!visited[u])
			dfs(u);
	}
	ans.pb(v);
}

void topological_sort() {
	visited.assign(n, false);
	ans.clear();
	for (int i = 0; i < n; ++i) {
		if (!visited[i])
			dfs(i);
	}
	reverse(ans.begin(), ans.end());
}

//topological sort
//with in degree to check cycle directed graph
//	kahn s algorithm
//	whenever we delete one edge indegree reduces by 1

queue<int>que;
vi a[maxn];
int indeg[maxn], indeg1[maxn];
int n, m, u, v, flag;
bool temp;
bool topsort()
{
	while (!que.empty())
		que.pop();
	for (int i = 1; i <= n; i++)
	{
		if (!indeg[i])
			que.push(i);
	}
	int now, num = 0;
	while (!que.empty())
	{
		now = que.front();
		que.pop();
		ans.pb(now);
		num++;
		for (int i = 0; i < a[now].size(); i++)
		{
			if (--indeg[a[now][i]] == 0)
				que.push(a[now][i]);
		}
	}
	if (num == n)
		return true;
	return false;
}



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

// Function to compute a/b under modlo m
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
//simplifies x/y
pair<int, int> simplify_fraction(int x, int y)
{
	int g = __gcd(x, y);
	x /= g, y /= g;
	if (x < 0)
		x = -x, y = -y;
	return {x, y};
}
int mul(int a, int b) {
	return ((a % M) * (b % M)) % M;
}
int sub(int a, int b) {
	return ((a % M) - (b % M) + M) % M;
}
int add(int a, int b) {
	a += b;
	if (a >= M)
		a -= M;
	return a;
}
int invM(int x) {
	return powM(x, M - 2);
}
void binomialCoeff(int n)
{
	f[0] = b[0] = 1;
	rep(i, 1, n) {
		f[i] = mul(f[i - 1], i);
		b[i] = invM(f[i]);
	}
	rep(i, 1, n) {
		C[i] = mul(mul(f[n], b[n - i]), b[i]);
	}
}
int binomialCoeffUtil(int n, int k, int** dp)
{
// If value in lookup table then return
	if (dp[n][k] != -1) //
		return dp[n][k];

// store value in a table before return
	if (k == 0) {
		dp[n][k] = 1;
		return dp[n][k];
	}

// store value in table before return
	if (k == n) {
		dp[n][k] = 1;
		return dp[n][k];
	}

// save value in lookup table before return
	dp[n][k] = binomialCoeffUtil(n - 1, k - 1, dp) +
	           binomialCoeffUtil(n - 1, k, dp);
	return dp[n][k];
}

int binomialCoeff(int n, int k)
{
	int** dp; // make a temporary lookup table
	dp = new int*[n + 1];

// loop to create table dynamically
	for (int i = 0; i < (n + 1); i++) {
		dp[i] = new int[k + 1];
	}

// nested loop to initialise the table with -1
	for (int i = 0; i < (n + 1); i++) {
		for (int j = 0; j < (k + 1); j++) {
			dp[i][j] = -1;
		}
	}

	return binomialCoeffUtil(n, k, dp);
}
int binomialCoeff(int n, int k)
{
	int C[k + 1];
	memset(C, 0, sizeof(C));

	C[0] = 1;  // nC0 is 1

	for (int i = 1; i <= n; i++)
	{
// Compute next row of pascal triangle using
// the previous row
		for (int j = min(i, k); j > 0; j--)
			C[j] = C[j] + C[j - 1];
	}
	return C[k];
}
int mul(int a, int b) {
	return ((a) * (b)) % M;
}
int add(int a, int b) {
	a += b;
	if (a >= M)a -= M;
	return a;
}
vvi p(vvi a, vvi b) {
	vvi x(n, vi(n, 0));
	rep(i, 0, n - 1)
	rep(j, 0, n - 1)
	rep(k, 0, n - 1)
	x[i][j] = add(x[i][j], mul(a[i][k], b[k][j]));
	return x;
}
vvi a;
//this return pow(A,x) where A is matrix
vvi matrix_power_final(vvi A, int x) {
	vvi result(n, vi(n, 0));
	rep(i, 0, n - 1)result[i][i] = 1;
	while (x) {
		if (x & 1)result = p(result , A);
		A = p(A , A);
		x = x / 2;
	}
	return result;
}
vvi p(vvi a, vvi b) {
	vvi x(a.size(), vi(b[0].size()));
	rep(i, 0, a.size() - 1)
	rep(j, 0, b[0].size() - 1)
	rep(k, 0, a[0].size() - 1)
	x[i][j] = add(x[i][j], mul(a[i][k], b[k][j]));
	return x;
}
vvi a;
vvi mat[70];
//this return pow(A,x) where A is matrix
vvi matrix_power_final(int s, int x) {
	vvi result(1, vi(n, 0)), A = a;
	result[0][s - 1] = 1;
	int pow = 2;
	while (x) {
		if (x & 1)result = p(result , A);
		A = mat[pow++];
		x = x / 2;
	}
	return result;
}
std::vi bfs(int s) {
	std::vi r;
	u[s] = 1;
	queue<int> q;
	q.push(s);
	while (!q.empty()) {
		int v = q.front();
		r.pb(v); q.pop();
		for (auto to : a[v]) {
			if (u[to] == 0) {
				u[to] = 1;
				q.push(to);
			}
		}
	}
	return r;
}
void bfs1(int s) {
	queue<int> qu;
	fill(vis, 0);

	qu.push(s);
	vis[s] = 1;
	dt[s]  = 0;

	while (!qu.empty()) {
		int c = qu.front();
		qu.pop();

		for (auto x : adj1[c]) {
			if (!vis[x]) {
				qu.push(x);
				vis[x] = 1;
				dt[x] = dt[c] + 1;
			}
		}
	}
}
int n;
int A[N], e[N][N];
std::vi v;
void dfs(int x) {
	if ((int)v.size() > k)return;
// if(u[x])return;
	u[x] = 1;
	v.pb(x);
// if(st)c[x]=i+1;
	for (int y : a[x])if (!u[y])
			dfs(y);
}
bool is(string str)
{
// Start from leftmost and rightmost corners of str
	int l = 0;
	int h = str.length() - 1;

// Keep comparing characters while they are same
	while (h > l)
	{
		if (str[l++] != str[h--])
		{

			return 0;
		}
	}
	return 1;
}
int parent[N], ra[N];
//disjoint set union dsu
int find_set(int v) {
	if (v == parent[v])
		return v;
	return parent[v] = find_set(parent[v]);
}
void make_set(int v) {
	parent[v] = v;
	ra[v] = 0;
}
void union_sets(int a, int b) {
	a = find_set(a);
	b = find_set(b);
	if (a != b) {
		if (ra[a] < ra[b])
			swap(a, b);
		parent[b] = a;
		if (ra[a] == ra[b])
			ra[a]++;
	}
}

//fast fourier transform fft
using cd = complex<double>;
const double PI = acos(-1);
int n;
void fft(vector<cd> & a, bool invert) {
	int n = a.size();

	for (int i = 1, j = 0; i < n; i++) {
		int bit = n >> 1;
		for (; j & bit; bit >>= 1)
			j ^= bit;
		j ^= bit;

		if (i < j)
			swap(a[i], a[j]);
	}

	for (int len = 2; len <= n; len <<= 1) {
		double ang = 2 * PI / len * (invert ? -1 : 1);
		cd wlen(cos(ang), sin(ang));
		for (int i = 0; i < n; i += len) {
			cd w(1);
			for (int j = 0; j < len / 2; j++) {
				cd u = a[i + j], v = a[i + j + len / 2] * w;
				a[i + j] = u + v;
				a[i + j + len / 2] = u - v;
				w *= wlen;
			}
		}
	}

	if (invert) {
		for (cd & x : a)
			x /= n;
	}
}
vi multiply(vi const& a, vi const& b) {
	vector<cd> fa(a.begin(), a.end()), fb(b.begin(), b.end());
	int n = 1;
	while (n < a.size() + b.size())
		n <<= 1;
	fa.resize(n);
	fb.resize(n);

	fft(fa, false);
	fft(fb, false);
	for (int i = 0; i < n; i++)
		fa[i] *= fb[i];
	fft(fa, true);

	vi result(n);
	for (int i = 0; i < n; i++)
		result[i] = round(fa[i].real());
	return result;
}
//fft with modulo
int modinv(int x) { return powM(x, M - 2); }
void fft(vi & a, bool invert) {
	int n = a.size();
	int root = powM(3, (M - 1) / n);
	int root_1 = modinv(root);

	for (int i = 1, j = 0; i < n; i++) {
		int bit = n >> 1;
		for (; j & bit; bit >>= 1)
			j ^= bit;
		j ^= bit;

		if (i < j)
			swap(a[i], a[j]);
	}

	for (int len = 2; len <= n; len <<= 1) {
		int wlen = invert ? root_1 : root;
		for (int i = len; i < n; i <<= 1)
			wlen = wlen * wlen % M;

		for (int i = 0; i < n; i += len) {
			int w = 1;
			for (int j = 0; j < len / 2; j++) {
				int u = a[i + j], v = a[i + j + len / 2] * w % M;
				a[i + j] = u + v < M ? u + v : u + v - M;
				a[i + j + len / 2] = u - v >= 0 ? u - v : u - v + M;
				w = w * wlen % M;
			}
		}
	}

	if (invert) {
		int n_1 = modinv(n);
		for (int & x : a)
			x = x * n_1 % M;
	}
}

vi multiply(vi const& a, vi const& b) {
	vi fa(a.begin(), a.end()), fb(b.begin(), b.end());
	int n = 1;
	while (n < a.size() + b.size())
		n <<= 1;
	fa.resize(n);
	fb.resize(n);


	fft(fa, false);
	fft(fb, false);
	for (int i = 0; i < n; i++)
		fa[i] = fa[i] * fb[i] % M;
	fft(fa, true);

	return fa;
}
int lcm(int a, int b)
{
	return (a * b) / __gcd(a, b);
}
int powM(int b, int p) {
	int r = 1;
	for (; p; b = b * b % M, p /= 2)
		if (p & 1)
			r = r * b % M;
	return r;
}
bool p[N];
pii inv_gcd(int a, int b) {
	a = ((a + b) % b);
	if (a == 0) return {b, 0};
	int s = b, t = a;
	int m0 = 0, m1 = 1;
	while (t) {
		int u = s / t;
		s -= t * u;
		m0 -= m1 * u;
		auto tmp = s;
		s = t;
		t = tmp;
		tmp = m0;
		m0 = m1;
		m1 = tmp;
	}
	if (m0 < 0) m0 += b / s;
	return pii{s, m0};
}
//chinese remainder theorem solution crt solution
pii crt_solution(const vi &r, const vi &m) {
	assert(r.size() == m.size());
	int n = (int)(r.size());
	int r0 = 0, m0 = 1;
	for (int i = 0; i < n; i++) {
		assert(1 <= m[i]);
		int r1 = ((r[i] + m[i]) % m[i]), m1 = m[i];
		if (m0 < m1) {
			std::swap(r0, r1);
			std::swap(m0, m1);
		}
		if (m0 % m1 == 0) {
			if (r0 % m1 != r1) return {0, 0};
			continue;
		}
		int g, im;
		std::tie(g, im) = inv_gcd(m0, m1);
		int u1 = (m1 / g);
		if ((r1 - r0) % g) return pii{0, 0};
		int x = (r1 - r0) / g % u1 * im % u1;
		r0 += x * m0;
		m0 *= u1;
		if (r0 < 0) r0 += m0;
	}
//ans, lcm of all m s
//0, 0 is returned if no solution
	return pii{r0, m0};
}
// int powM(int a,int b)
// {
//   int ans=1;
//   while(b)
//   {
//     if(b&1LL)ans=ans*a%M;
//     a=a*a%M;
//     b>>=1;
//   }
//   return ans;
// }
//spf
void Sieve() {
	rep(i, 2, N - 1) {
		if (sp[i] == 0) {
			for (int j = i; j < N; j += i) {
				sp[j] = i;
			}
		}
	}
	rep(i, 2, N - 1) {
		koi[i] = koi[i / sp[i]] + (sp[i / sp[i]] != sp[i]);
		// koi is number of prime factors of i
	}
}
vi fact(int n) {
	vi v;
	for (int i = 1; i * i <= n; ++i) {
		if (n % i == 0) {
			v.pb(i);
			if (i * i != n) {
				v.pb(n / i);
			}
		}
	}
	return v;
}
int LIS(int l, int r, int mi = -inf, int ma = inf) {
	vi lis;
// make both < = instead of < if increasing ans not strictly increasing
//optionally min and max can be passed
	for (int j = l; j <= r; ++j) if (mi <= a[j] && a[j] <= ma) {
			auto pos = upper_bound(lis.begin(), lis.end(), a[j]);
			if (pos == lis.end()) lis.pb(a[j]);
			else *pos = a[j];
		}
	return (lis.size());
}
//finds fibo(n) in log n
map<int, int> F;
int f(int n) {
//initialize F[0],F[1],F[2] before calling this function;
	if (F.count(n)) return F[n];
	int k = n / 2;
	if (n % 2 == 0) { // n=2*k
		return F[n] = (f(k) * f(k) + f(k - 1) * f(k - 1)) % M;
	} else { // n=2*k+1
		return F[n] = (f(k) * f(k + 1) + f(k - 1) * f(k)) % M;
	}
}
int div(int n)
{
	int d[1000001] = {0};
	for (int i = 1; i <= 1000000; i++)
		for (int j = i; j <= 1000000; j += i)
			d[j]++;
	return d[n];
}
double dist(pair<double, double>a, pair<double, double>b) {
	return sqrt((a.first - b.first) * (a.first - b.first) + (a.second - b.second) * (a.second - b.second));
}
//next greater
//this works for next greater than element
fill(l, l + n, - 1);
fill(r, r + n, - 1);
stack<int>s;
for (int i = n - 1; i > -1; i--) {
	while (s.empty() == 0 and a[s.top()] <= a[i])s.pop();
	if (s.empty() == 0)r[i] = s.top();
	s.push(i);
}
s = stack<int>();
for (int i = 0; i < n; i++) {
	while (s.empty() == 0 and a[s.top()] <= a[i])s.pop();
	if (s.empty() == 0)l[i] = s.top();
	s.push(i);
}
int f[N], b[N];
int mul(int a, int b) {
	return (a * b) % M;
}
int add(int a, int b) {
	a += b;
	if (a >= M)
		a -= M;
	return a;
}
int sub(int a, int b) {
	return (a - b + M) % M;
}
int powM(int b, int p) {
	int r = 1;
	for (; p; b = b * b % M, p /= 2) {
		if (p & 1)
			r = r * b % M;
	}
	return r;
}
int invM(int x) {
	return powM(x, M - 2);
}
void binomialCoeff() {
	f[0] = b[0] = f[1] = b[1] = 1;
	rep(i, 2, N - 1) {
		f[i] = mul(f[i - 1], i);
		b[i] = mul(M / i, b[M % i]);
	}
}
int C(int n, int r) {
	assert(n < N); assert(r < N);
	if (r > n)return 0;
	return (f[n] * mul(b[r], b[n - r]) % M) % M;
}
void binomialCoeff() {
	f[0] = b[0] = 1;
	rep(i, 1, N - 1) {
		f[i] = mul(f[i - 1], i);
		b[i] = powM(f[i], M - 2);
	}
}
vvi prime_vector_sieve() {
	vvi prime(N);
	for (int i = 2; i < N; ++i)if (prime[i].empty()) {
			for (int j = i; j < N; j += i) {
				prime[j].pb(i);
			}
		}
	return prime;
}
void binary_lifting() {
	for (int i = 1; i <= n; i++) {
		f[i] = f[i - 1];
		for (int j = 2; j * j <= a[i]; ++j)
			if (a[i] % j == 0) {
				f[i] = max(f[i], lst[j]);
				lst[j] = i;
				while (a[i] % j == 0)a[i] /= j;
			}
		if (a[i] ^ 1) {
			f[i] = max(f[i], lst[a[i]]);
			lst[a[i]] = i;
		}
	}
	for (int i = 1; i <= n; ++i)g[0][i] = f[i];
	for (int i = 1; i <= 20; ++i)for (int j = 1; j <= n; ++j)g[i][j] = g[i - 1][g[i - 1][j]];
}
int mex() {
	map<int, bool> m;
	for (int i = 0; i < n; i++) m[a[i]] = true;
	for (int i = 0; i < n; i++) {
		if (!m[i]) return i;
	}
	return n;
}
long long C(long long n, long long r)
{
	long long N = 1, R = 1;
	if (r > n or r < 0)return 0;
	if (n == r or r == 0)return 1;
	for (long long i = 1; i <= n; i++) N = N * i % M;
	for (long long i = 1; i <= r; i++) R = R * i % M;
	for (long long i = 1; i <= n - r; i++) R = R * i % M;
	return (N * powM(R, M - 2)) % M;
}
int fact_dp(int n)

{

	if (n >= 0)

	{

		result[0] = 1;

		for (int i = 1; i <= n; ++i)

		{

			result[i] = i * result[i - 1];

		}

		return result[n];

	}

}
int binarySearch(int arr[], int l, int r, int x, int n)
{
	while (l <= r) {
		int m = l + (r - l) / 2;

// Check if x is present at mid
		if ((arr[m] <= x) and ( (arr[m + 1] > x) or (m + 1 == n)) /*and (m>0)*/)
			return m;

// If x greater, ignore left half
		if (arr[m] < x)
			l = m + 1;

// If x is smaller, ignore right half
		else
			r = m - 1;
	}

// if we reach here, then element was
// not present
	return -1;
}
void precompute(int s[], int n, int l[][MAX],
                int r[][MAX])
{
	l[s[0]][0] = 1;

// Precompute the prefix 2D array
	for (int i = 1; i < n; i++) {
		for (int j = 0; j < MAX_CHAR; j++)
			l[j][i] += l[j][i - 1];

		l[s[i]][i]++;
	}

	r[s[n - 1]][n - 1] = 1;

// Precompute the Suffix 2D array.
	for (int i = n - 2; i >= 0; i--) {
		for (int j = 0; j < MAX_CHAR; j++)
			r[j][i] += r[j][i + 1];

		r[s[i]][i]++;
	}
}
// Find the number of palindromic subsequence of
// length k
int countPalindromes(int k, int n, int l[][MAX],
                     int r[][MAX])
{
	int ans = 0;

// If k is 1.
	if (k == 1) {
		for (int i = 0; i < MAX_CHAR; i++)
			ans += l[i][n - 1];
		return ans;
	}

// If k is 2
	if (k == 2) {

// Adding all the products of prefix array
		for (int i = 0; i < MAX_CHAR; i++)
			ans += ((l[i][n - 1] * (l[i][n - 1] - 1)) / 2);
		return ans;
	}

// For k greater than 2. Adding all the products
// of value of prefix and suffix array.
	for (int i = 1; i < n - 1; i++)
		for (int j = 0; j < MAX_CHAR; j++)
			ans += l[j][i - 1] * r[j][i + 1];

	return ans;
}
//cycle detecttion in undirectedgraph
void dfs(int i, int p = 0) {
	vis[i] = 1;
	s.push(i);
	D[i] = s.size();
	for (auto x : a[i]) {
		if (vis[x] == 0) {
			dfs(x, i);
		}
		else {
			if (D[i] - D[x] + 1 <= k and D[i] - D[x] + 1 >= 3) {
//enter a cycle
				cout << 2 << "\n";
				cout << D[i] - D[x] + 1 << "\n";
				int z = D[i] - D[x] + 1;
				while (z--)
				{cout << s.top() << " "; s.pop();}
				exit(0);
			}
		}
	}
	for (auto x : a[i]) {
		if (d.count(i) == 0)d.insert(x);
	}
	s.pop();
}
// const int N1 = 1e5;  // limit for array size
// int n;  // array size
// int t[2 * N1];

// void build() {  // build the tree
//   for (int i = n - 1; i > 0; --i) t[i] = t[i<<1] + t[i<<1|1];
// }

// void modify(int p, int value) {  // set value at position p
//   for (t[p += n] = value; p > 1; p >>= 1) t[p>>1] = t[p] + t[p^1];
// }

// int query(int l, int r) {  // sum on interval [l, r)
//   int res = 0;
//   for (l += n, r += n; l < r; l >>= 1, r >>= 1) {
//     if (l&1) res += t[l++];
//     if (r&1) res += t[--r];
//   }
//   return res;
// }
// int A[N],r[N];
// bool cmp(int a, int b) {
//     return(A[a] == A[b] ? (a < b) : (A[a] > A[b]));
// }
//returns number of elements with nth bit set in v
void solve() {
// cin>>n;
// for (int i = 0; i < n; ++i)
// {
//     cin>>t[n+i];
// }
// build();
// string s;cin>>s;std::map<char, set<char> > m;
// for (int i = 0; i < s.length(); ++i)
// {
//     if(i==0){m[s[i]].insert(s[i+1]);continue;}
//     if(i==s.length()-1){m[s[i]].insert(s[i-1]);continue;}
//     m[s[i]].insert(s[i-1]),m[s[i]].insert(s[i+1]);
// }
// for(auto i: m){
//     for (auto j:i.S)
//     {
//         // cout<<j<<" ";
//         {if(!m[j].count(i.F))cout<<"NO\n";return;}
//     }
//     cout<<"\n";
// }
}
int main()
{
	ibs; cti;
// int t;cin>>t;
// while(t--)
	solve(), cout << "\n";
	return 0;
}