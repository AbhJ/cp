 /**
 * @author      : abhj
 * @created     : {{TIMESTAMP}}
 * @filename    : {{FILE}}.cpp
 */

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
using namespace std;
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
template<typename T>
using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
const int N    =     2e6 + 10;

int n, a[N]; long long p[N], t, ans;
void solve() {
    ordered_set<pair<long long, int>>s;
    cin >> n >> t;
    // INITIALLY THERE WAS NO SUM
    s.insert(mp(0, 0));
    for(int i = 0; i < n; i++){
        cin >> a[i];
        p[i] = a[i] + (i == 0 ? 0 : p[i - 1]); 

        ans += s.size() - s.order_of_key(mp(p[i] - t, 1e9));

        s.insert(mp(p[i], i));

        // s[i] - s[x] < t
        // WE WANT TO REMOVE NUMBER OF ELEMENTS s[x] >= s[i] - t
        // ORDER OF KEY GIVES COUNT OF LOWER ELEMENTS
        // ALSO, WE WANT TO HAVE A MULTISET THAN A SET
        // TO INTRODUCE THAT, WE ADD INDEX VALUES TO MAKE
        // ORDERED SET UNIQUE

    }
    cout << ans;
}

int32_t main() {
	ios_base::sync_with_stdio(false); cin.tie(0);
	solve();
	return 0;
}
