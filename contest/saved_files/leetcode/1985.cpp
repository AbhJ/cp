class Solution {
public:

	string cleaned (string s) {
		int cnt = 0;
		while (s.length() != cnt + 1 and s[cnt] == '0')
			cnt++;
		return s.substr (cnt);
	}

	static bool cmp (string& a, string& b) {
		if (a.size() == b.size())
			return a < b;
		return a.size() < b.size();
	}

	string helper (vector<string>& nums, int k) {
		nth_element (nums.begin(), nums.end() - k, nums.end(), cmp);
		return nums[nums.size() - k];
	}

	string kthLargestNumber (vector<string>& nums, int k) {
		if (nums.size() > 5e3)
			return helper (nums, k);
		for (auto &s : nums)
			s = string (1e4 - s.length(), '0') + s;
		multiset<string> s (nums.begin(), nums.end());
		multiset<string>::iterator itr = s.begin();
		advance (itr, nums.size() - k);
		return cleaned (*itr);
	}
};
