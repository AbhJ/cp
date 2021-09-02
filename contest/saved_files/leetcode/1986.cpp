
class Solution {
public:

	int helper (vector<int>& tasks, int sessionTime) {
		int i = 0, j = tasks.size(), koi = 0;
		while (i < j and ++koi) {
			int cnt = tasks[i];
			while (i < j - 1 and cnt + tasks[i + 1] <= sessionTime)
				cnt += tasks[++i];
			while (j - 1 > i and cnt + tasks[j - 1] <= sessionTime)
				cnt += tasks[--j];
			if (j - 1 == i)
				return koi;
			++i;
		}
		return 0;
	}

	int ans = 1e9;

	int helper_brute (vector<int>& tasks, int sessionTime) {
		vector<int>sess;
		ans = 1e9;
		function<void (vector<int>&, int, vector<int>&, int)> f =
		[&] (vector<int>& tasks, int sessionTime, vector<int>& sess, int pos) {
			if (sess.size() > ans)
				return;
			if (pos == tasks.size()) {
				ans = min (ans, (int)sess.size());
				return;
			}
			// ADD TO EXISTING SESSION
			for (int i = 0; i < sess.size(); ++i) {
				if (sess[i] + tasks[pos] <= sessionTime) {
					sess[i] += tasks[pos];
					f (tasks, sessionTime, sess, pos + 1);
					sess[i] -= tasks[pos];
				}
			}
			// CREATE NEW SESSION FOR THIS JOB
			sess.emplace_back (tasks[pos]);
			f (tasks, sessionTime, sess, pos + 1);
			sess.pop_back();
		};
		f (tasks, sessionTime, sess, 0);
		return ans;
	}

	int minSessions (vector<int>& tasks, int sessionTime) {
		sort (tasks.begin(), tasks.end());
		reverse (tasks.begin(), tasks.end());
		return min (helper (tasks, sessionTime), helper_brute (tasks, sessionTime));
	}
};
