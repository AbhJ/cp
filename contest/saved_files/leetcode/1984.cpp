class Solution {
public:
	int minimumDifference (vector<int>& nums, int k) {
		sort (nums.begin(), nums.end());
		assert (k <= nums.size());
		int ans = 1e5;
		for (int i = k - 1; i < nums.size(); i++)
			ans = min (ans, nums[i] - nums[i - k + 1]);
		return ans;
	}
};
