class Solution {
public:
	int numberOfUniqueGoodSubsequences (string binary) {
		int ones (0), zeros (0), dp[2] {0};
		for (int i = 0; i < binary.length(); i++) {
			if (binary[i] == '0') {
				zeros++;
				dp[0] = dp[1] + dp[0];
				// ZERO CANNOT INITIATE STRING
			}
			else {
				ones++;
				dp[1] = dp[1] + dp[0] + 1;
				// ONE CAN INITIATE STRINGS
			}
			dp[0] %= (int)(1e9 + 7);
			dp[1] %= (int)(1e9 + 7);
		}
		return (dp[0] + dp[1] + !!zeros) % (int)(1e9 + 7);
	}
};
