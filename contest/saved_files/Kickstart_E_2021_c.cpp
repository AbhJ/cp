

/**
 * @author      : abhj
 * @created     : Sunday Aug 22, 2021 10:04:10 IST
 * @filename    : c.cpp
 */

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
const int N    =     1e3 + 10;

char arr[N][N];
int n, m, cn;

int ok(int i, int j) {
	return arr[i][j] >= 'A' and arr[i][j] <= 'Z';
}

void outp() {
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= m; j++) {
			cout << arr[i][j];
		}
		cout << "\n";
	}
}

void cnt(int& res) {
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= m; j++) {
			if (ok(i, j)) {
				res++;
			}
		}
	}

}

void print_output() {

	int res = 0;

	cnt(res);

	cout << -cn + res << "\n";
	outp();
}


void solve() {
	cin >> n >> m;
	vi R[n + 1], C[m + 1];
	cn = 0;
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= m; j++) {
			cin >> arr[i][j];
			if (ok(i, j))
				cn++;
		}
	}

	for (int i = 1; i <= n; i++) {
		R[i].pb(0);
		for (int j = 1; j <= m; j++)
			if (arr[i][j] == '#')
				R[i].pb(j);
		R[i].pb(m + 1);
	}


	for (int j = 1; j <= m; j++) {
		C[j].pb(0);
		for (int i = 1; i <= n; i++) {
			if (arr[i][j] == '#') {
				C[j].pb(i);
			}
		}
		C[j].pb(n + 1);
	}

	queue<pii> que;

	for (int i = 1; i <= n; i++)
		for (int j = 1; j <= m; j++)
			if (ok(i, j))
				que.emplace(i, j);

	while (que.empty() == 0) {
		auto zzz = que.front();
		que.pop();
		int I = zzz.F, J = zzz.S;
		int whic = upper_bound(R[I].begin(), R[I].end(), J) - R[I].begin();

		int yy = 2, po = R[I][whic] - 1 - (J - (R[I][whic - 1] + 1)), zz = 1;

		char ch = arr[I][J];

		if (arr[I][po] != '#' and !ok(I, po))
			arr[I][po] = ch,
						 que.emplace(I, po);

		whic = upper_bound(C[J].begin(), C[J].end(), I) - C[J].begin();
		po = C[J][whic] - 1 - (I - (C[J][whic - 1] + 1));

		if (arr[po][J] != '#' and !ok(po, J))
			arr[po][J] = ch,
						 que.emplace(po, J);

	}

	print_output();
}

int32_t main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	int t, I = 0;
	cin >> t;
	while (t--) {
		cout << "Case #" << ++I << ": ";
		solve();
	}
	return 0;
}
