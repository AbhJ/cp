#include "bits/stdc++.h"
// #define int          long long int
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

// const int inf  =     1e18 + 10;
const int N    =     2e5 + 10;
int n, a[N], b[N], k[N], pos[N], m, ans[N]; vpii q[N];
int t[4 * N];
struct Vertex {
	Vertex *l, *r;
	int sum;

	Vertex(int val) : l(nullptr), r(nullptr), sum(val) {}
	Vertex(Vertex *l, Vertex *r) : l(l), r(r), sum(0) {
		if (l) sum += l->sum;
		if (r) sum += r->sum;
	}
};
Vertex* build(int tl, int tr) {
	if (tl == tr)
		return new Vertex(0);
	int tm = (tl + tr) / 2;
	return new Vertex(build(tl, tm), build(tm + 1, tr));
}

Vertex* update(Vertex* v, int tl, int tr, int pos) {
	if (tl == tr)
		return new Vertex(v->sum + 1);
	int tm = (tl + tr) / 2;
	if (pos <= tm)
		return new Vertex(update(v->l, tl, tm, pos), v->r);
	else
		return new Vertex(v->l, update(v->r, tm + 1, tr, pos));
}

int find_kth(Vertex* vl, Vertex *vr, int tl, int tr, int k) {
	if (tl == tr)
		return tl;
	int tm = (tl + tr) / 2, left_count = vr->l->sum - vl->l->sum;
	if (left_count >= k)
		return find_kth(vl->l, vr->l, tl, tm, k);
	return find_kth(vl->r, vr->r, tm + 1, tr, k - left_count);
}
static bool cmp(pii x, pii y) {
	if (x.F != y.F)
		return x.F > y.F;
	return x.S < y.S;
}
vpii v;
void solve() {
	cin >> n;
	v.resize(n);
	for (int i = 1; i <= n; i++) {
		cin >> a[i];
		v[i - 1] = mp(a[i], i);
	}
	cin >> m;
	sort(v.begin(), v.end(), cmp);
	std::vector<Vertex*> roots;
	roots.pb(build(0, n));
	for (int i = 1; i <= m; i++) {
		cin >> k[i] >> pos[i];
		q[k[i]].pb(pos[i], i);
	}
	for (int i = 1; i <= n; i++) {
		roots.pb(update(roots.back(), 0, n, v[i - 1].S));
		for (auto [posi, inde] : q[i]) {
			ans[inde] = find_kth(roots[0], roots[i], 0, n, posi);;
		}
	}
	for (int i = 1; i <= m; i++) {
		cout << a[ans[i]] << "\n";
	}

}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	solve(); return 0;
	return 0;
}