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
#define rep(i, a, b) for (int i = a; i <= b; i++)
#define per(i, b, a) for (int i = b; i >= a; i--)
#define all(x)       x.begin(), x.end()
using namespace std;
const int inf  =     1e18 + 10;
const int N    =     2e6 + 10;
int k, n, ans, tmp, q, p[N], dp[N]; char c[N]; string s;
void solve() {
    cin >> k >> s;
    cin >> q;
    rep(i, 1, q) {
        cin >> p[i] >> c[i];
    }
    fill(dp, dp + N, 1);
    reverse(all(s));
    per(i, (1 << k) - 2, 0) {
        if (s[i] == '0')
            dp[i] = dp[(i << 1) + 2];
        if (s[i] == '1')
            dp[i] = dp[(i << 1) + 1];
        if (s[i] == '?')
            dp[i] = dp[(i << 1) + 1] + dp[(i << 1) + 2];
    }
    rep(i, 1, q) {
        p[i] = (1 << k) - p[i] - 1;
        s[p[i]] = c[i];
        // 2 to 8
        while (p[i] > - 1) {
            if (s[p[i]] == '0')
                dp[p[i]] = dp[(p[i] << 1) + 2];
            if (s[p[i]] == '1')
                dp[p[i]] = dp[(p[i] << 1) | 1];
            if (s[p[i]] == '?')
                dp[p[i]] = dp[(p[i] << 1) | 1] + dp[(p[i] << 1) + 2];
            if (p[i] == 0)
                break;
            p[i] --;
            p[i] /= 2;
        }
        cout << dp[0] << "\n";
    }
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