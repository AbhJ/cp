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
int n, m, A[N], B[N], vis[N], cyc[N], dou[N], col[N]; vvi a;

void dfs(int u) {
	vis[u] = 1;
	for (auto &i : a[u]) {
		if (vis[i] == 1) {
			cyc[i] = 1;
		}
		if (vis[i] == 2) {
			dou[i] = 1;
		}
		if (vis[i] == 0) {
			vis[i] = 1;
			dfs(i);
		}
	}
	vis[u] = 2;
}

struct BFS {
	vi distance;
	vi parent;

	BFS(const vvi &adj, const vi &sources) {
		const int n = adj.size();
		distance.assign(n, -1);
		parent.assign(n, -1);
		queue<int> q;
		for (int i : sources) {
			q.push(i);
			distance[i] = 0;
		}
		while (!q.empty()) {
			int i = q.front();
			q.pop();
			for (int j : adj[i]) {
				if (distance[j] != -1)
					continue;

				q.push(j);
				distance[j] = distance[i] + 1;
				parent[j] = i;
			}
		}
	}

	vi get_path(int destination) {
		if (distance[destination] == -1)
			return {};
		vi path;
		for (int i = destination; i != -1; i = parent[i])
			path.pb(i);
		reverse(path.begin(), path.end());
		return path;
	}
};
void solve() {
	cin >> n >> m;
	a.assign(n + 1, vi());
	fill(cyc + 1, cyc + n + 1, 0LL);
	fill(dou + 1, dou + n + 1, 0LL);
	fill(col + 1, col + n + 1, 0LL);
	fill(vis + 1, vis + n + 1, 0LL);
	for (int i = 1; i <= m; i++) {
		cin >> A[i] >> B[i];
		a[A[i]].pb(B[i]);
	}
	dfs(1);
	vi cyc_sources;
	vi dou_sources;
	for (int i = 1; i <= n; i++) {
		if (cyc[i] == 1) {
			cyc_sources.pb(i);
			// cout << i << " ";
		}
		else if (dou[i] == 1) {
			dou_sources.pb(i);
		}
		if (vis[i] == 2) {
			col[i] = 1;
		}
	}
	BFS cyc_bfs(a, cyc_sources);
	BFS dou_bfs(a, dou_sources);
	for (int i = 1; i <= n; i++) {
		if (cyc_bfs.distance[i] != - 1) {
			col[i] = - 1;
		}
		else if (dou_bfs.distance[i] != - 1) {
			col[i] = 2;
		}
		cout << col[i] << " ";
	}
}
int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	int t; cin >> t;
	while (t--) {
		solve(); cout << "\n";
	}
	return 0;
}