package categories;

import java.util.Set;

public class DynamicProgramming {

    /**
     * Unique Paths.
     *
     * A robot is located at the top-left corner of a m x n grid (marked 'Start'
     * in the diagram below). The robot can only move either down or right at
     * any point in time. The robot is trying to reach the bottom-right corner
     * of the grid (marked 'Finish' in the diagram below). How many possible
     * unique paths are there?
     *
     * Notice: m and n will be at most 100.
     *
     * @param n,
     *            m: positive integer (1 <= n ,m <= 100)
     * @return an integer
     */
    @tags.DynamicProgramming
    @tags.Array
    public int uniquePaths(int m, int n) {
        if (m == 0 || n == 0) {
            return 0;
        }

        int[][] matrix = new int[m][n];
        matrix[0][0] = 1;
        for (int i = 0, j = 1; j < n; j++) {
            matrix[i][j] = matrix[i][j - 1];
        }
        for (int i = 1, j = 0; i < m; i++) {
            matrix[i][j] = matrix[i - 1][j];
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                matrix[i][j] = matrix[i - 1][j] + matrix[i][j - 1];
            }
        }

        return matrix[m - 1][n - 1];
    }

    /**
     * Unique Paths (1 dimensional Array).
     */
    public int uniquePaths1D(int m, int n) {
        if (m == 0 || n == 0) {
            return 0;
        }

        int[] dp = new int[n];
        dp[0] = 1;

        for (int i = 0; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[j] += dp[j - 1];
            }
        }

        return dp[n - 1];
    }

    /**
     * Unique Paths II
     *
     * Follow up for "Unique Paths": Now consider if some obstacles are added to
     * the grids. How many unique paths would there be? An obstacle and empty
     * space is marked as 1 and 0 respectively in the grid.
     *
     * Notice: m and n will be at most 100.
     *
     * Additional obstacle matrix
     */
    @tags.DynamicProgramming
    @tags.Array
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid == null || obstacleGrid.length == 0 || obstacleGrid[0].length == 0) {
            return 0;
        }

        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int[] uniquePath = new int[n];

        // if there is no obstacle at left upper corner
        if (obstacleGrid[0][0] == 0) {
            uniquePath[0] = 1;
        }

        for (int i = 0; i < m; i++) {
            if (obstacleGrid[i][0] == 1) {
                uniquePath[0] = 0;
            }
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 1) {
                    uniquePath[j] = 0;
                } else {
                    uniquePath[j] += uniquePath[j - 1];
                }
            }
        }

        return uniquePath[n - 1];
    }

    /**
     * Climbing Stairs.
     *
     * You are climbing a stair case. It takes n steps to reach to the top. Each
     * time you can either climb 1 or 2 steps. In how many distinct ways can you
     * climb to the top?
     *
     * Too simple, it's just like Fibonacci, we can even make it O(logn) or
     * O(1).
     *
     * @param n:
     *            An integer
     * @return: An integer
     */
    @tags.DynamicProgramming
    public int climbStairs(int n) {
        // OJ doesn't test this case, should ask interviewer
        if (n == 0) {
            return 1;
        }

        // just like fibonacci
        int a = 1;
        int b = 1;

        while (n-- > 1) {
            int temp = a + b;
            a = b;
            b = temp;
        }

        return b;
    }

    /**
     * Minimum Path Sum.
     *
     * Given a m x n grid filled with non-negative numbers, find a path from top
     * left to bottom right which minimizes the sum of all numbers along its
     * path.
     *
     * Notice: You can only move either down or right at any point in time.
     *
     * @param grid:
     *            a list of lists of integers.
     * @return: An integer, minimizes the sum of all numbers along its path
     */
    @tags.DynamicProgramming
    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }

        int m = grid.length;
        int n = grid[0].length;
        int[] minPathSum = new int[n];

        // initialize row 0
        minPathSum[0] = grid[0][0];
        for (int j = 1; j < n; j++) {
            minPathSum[j] = minPathSum[j - 1] + grid[0][j];
        }

        for (int i = 1; i < m; i++) {
            minPathSum[0] += grid[i][0];
            for (int j = 1; j < n; j++) {
                minPathSum[j] = Math.min(minPathSum[j], minPathSum[j - 1]);
                minPathSum[j] += grid[i][j];
            }
        }

        return minPathSum[n - 1];
    }

    /**
     * Triangle.
     *
     * Given a triangle, find the minimum path sum from top to bottom. Each step
     * you may move to adjacent numbers on the row below.
     *
     * Space: O(n) Time: O(1)
     *
     * @param triangle:
     *            a list of lists of integers.
     * @return: An integer, minimum path sum.
     */
    @tags.DynamicProgramming
    public int minimumTotal(int[][] triangle) {
        // assume a triagle
        if (triangle == null || triangle.length == 0) {
            return 0;
        }

        int n = triangle.length;
        int[] minTotal = new int[n + 1];

        // do it bottom up to based on relationship between layers
        for (int i = n - 1; i >= 0; i--) {
            for (int j = 0; j <= i; j++) {
                minTotal[j] = Math.min(minTotal[j], minTotal[j + 1]);
                minTotal[j] += triangle[i][j];
            }
        }

        return minTotal[0];
    }

    /**
     * Jump Game
     *
     * Given an array of non-negative integers, you are initially positioned at
     * the first index of the array. Each element in the array represents your
     * maximum jump length at that position. Determine if you are able to reach
     * the last index.
     *
     * For example: A = [2,3,1,1,4], return true. A = [3,2,1,0,4], return false.
     *
     * NoticeThis problem have two method which is Greedy and Dynamic
     * Programming.
     *
     * The time complexity of Greedy method is O(n).
     *
     * The time complexity of Dynamic Programming method is O(n^2). However this
     * is, it is not straight forward.
     *
     * @param A:
     *            A list of integers
     * @return: The boolean answer
     */
    @tags.Greedy
    @tags.DynamicProgramming
    @tags.Array
    public boolean canJump(int[] A) {
        if (A == null || A.length < 2) {
            return true;
        }

        // farthest distance can be reached
        int distance = 0;

        // traverse A to update the reach
        for (int i = 0; i <= distance && i < A.length; i++) {
            distance = Math.max(distance, A[i] + i);
        }

        return distance >= A.length - 1;
    }

    /**
     * Jump Game II.
     *
     * Given an array of non-negative integers, you are initially positioned at
     * the first index of the array. Each element in the array represents your
     * maximum jump length at that position. Your goal is to reach the last
     * index in the minimum number of jumps.
     *
     * For example: Given array A = [2,3,1,1,4]. The minimum number of jumps to
     * reach the last index is 2. (Jump 1 step from index 0 to 1, then 3 steps
     * to the last index.)
     *
     * @param A:
     *            A list of lists of integers
     * @return: An integer
     */
    @tags.Greedy
    @tags.Array
    public int jump(int[] A) {
        if (A == null || A.length == 0) {
            return -1;
        }

        int[] jumps = new int[A.length];
        for (int i = 1; i < A.length; i++) {
            jumps[i] = Integer.MAX_VALUE;
        }

        int distance = 0;
        for (int i = 0; i < A.length; i++) {
            distance = Math.max(distance, i + A[i]);
            for (int j = i + 1; j <= distance && j < A.length; j++) {
                jumps[j] = Math.min(jumps[i] + 1, jumps[j]);
            }
        }

        if (distance < A.length - 1) {
            return -1;
        } else {
            return jumps[jumps.length - 1];
        }
    }

    public int jump2(int[] A) {
        int len = A.length;
        int ret = 0;
        int last = 0;
        int curr = 0;

        for (int i = 0; i < len; ++i) {
            if (i > last) {
                last = curr;
                ++ret;
            }

            curr = Math.max(curr, i + A[i]);
        }

        return ret;
    }

    /**
     * Longest Increasing Subsequence.
     *
     * Given a sequence of integers, find the longest increasing subsequence
     * (LIS). You code should return the length of the LIS.
     *
     * https://en.wikipedia.org/wiki/Longest_increasing_subsequence
     *
     * @param nums:
     *            The integer array
     * @return: The length of LIS (longest increasing subsequence)
     */
    @tags.BinarySearch
    @tags.DynamicProgramming
    @tags.Site.LintCode
    public int longestIncreasingSubsequence(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }

        int[] lisDP = new int[nums.length];
        int max = 1;

        for (int i = 0; i < nums.length; i++) {
            lisDP[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[j] <= nums[i]) {
                    lisDP[i] = Math.max(lisDP[i], lisDP[j] + 1);
                    max = Math.max(lisDP[i], max);
                }
            }
        }

        return max;
    }

    /**
     * nlogn solution. TODO: I don't understand yet.
     *
     * @param nums
     * @return
     */
    public int longestIncreasingSubsequence2(int[] nums) {
        int[] minLast = new int[nums.length + 1];
        minLast[0] = -1;
        for (int i = 1; i <= nums.length; i++) {
            minLast[i] = Integer.MAX_VALUE;
        }

        for (int i = 0; i < nums.length; i++) {
            // find the first number in minLast > nums[i]
            int index = binarySearch(minLast, nums[i]);
            minLast[index] = nums[i];
        }

        for (int i = nums.length; i >= 1; i--) {
            if (minLast[i] != Integer.MAX_VALUE) {
                return i;
            }
        }

        return 0;
    }

    // find the first number > num
    private int binarySearch(int[] minLast, int num) {
        int start = 0, end = minLast.length - 1;
        while (start + 1 < end) {
            int mid = (end - start) / 2 + start;
            if (minLast[mid] == num) {
                start = mid;
            } else if (minLast[mid] < num) {
                start = mid;
            } else {
                end = mid;
            }
        }

        if (minLast[start] > num) {
            return start;
        }
        return end;
    }

    /**
     * Coin Change.
     *
     * You are given coins of different denominations and a total amount of
     * money amount. Write a function to compute the fewest number of coins that
     * you need to make up that amount. If that amount of money cannot be made
     * up by any combination of the coins, return -1.
     *
     * Example 1: coins = [1, 2, 5], amount = 11 return 3 (11 = 5 + 5 + 1)
     *
     * Example 2: coins = [2], amount = 3 return -1.
     *
     * Note: You may assume that you have an infinite number of each kind of
     * coin.
     *
     * @param coins
     * @param amount
     * @return
     */
    @tags.DynamicProgramming
    @tags.Site.LeetCode
    public int coinChange(int[] coins, int amount) {
        if (coins == null || coins.length == 0) {
            return -1;
        }

        // initialize
        int[] count = new int[amount + 1];
        for (int i = 1; i < count.length; i++) {
            count[i] = -1;
        }
        for (Integer coin : coins) {
            if (coin < count.length) {
                count[coin] = 1;
            }
        }

        for (int i = 1; i < count.length; i++) {
            for (Integer coin : coins) {
                if (i - coin >= 0) {
                    if (count[i] == -1 && count[i - coin] == -1) {
                        continue;
                    } else if (count[i] == -1) {
                        count[i] = count[i - coin] + 1;
                    } else if (count[i - coin] == -1) {
                        continue;
                    } else {
                        count[i] = Math.min(count[i - coin] + 1, count[i]);
                    }
                }
            }
        }

        return count[amount];
    }

    /**
     * Edit Distance
     *
     * Given two words word1 and word2, find the minimum number of steps
     * required to convert word1 to word2. (each operation is counted as 1
     * step.)
     *
     * You have the following 3 operations permitted on a word: Insert a
     * character Delete a character Replace a character
     *
     * @param word1
     *            & word2: Two string.
     * @return: The minimum number of steps.
     */
    @tags.String
    @tags.DynamicProgramming
    public int minDistance(String word1, String word2) {
        if (word1 == null || word2 == null)
            return -1; // invalid input
        int len1 = word1.length();
        int len2 = word2.length();
        int[][] dist = new int[len1 + 1][len2 + 1];

        // first row
        for (int i = 1; i <= len2; i++) {
            dist[0][i] = i;
        }

        // first column
        for (int i = 1; i <= len1; i++) {
            dist[i][0] = i;
        }

        // fill the remaining elements
        for (int i = 1; i <= len1; i++) {
            for (int j = 1; j <= len2; j++) {
                dist[i][j] = Math.min(dist[i - 1][j], dist[i][j - 1]) + 1;
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dist[i][j] = Math.min(dist[i][j], dist[i - 1][j - 1]);
                } else {
                    dist[i][j] = Math.min(dist[i][j], dist[i - 1][j - 1] + 1);
                }
            }
        }

        return dist[len1][len2];
    }

    /**
     * Distinct Subsequences
     *
     * Given a string S and a string T, count the number of distinct
     * subsequences of T in S.
     *
     * A subsequence of a string is a new string which is formed from the
     * original string by deleting some (can be none) of the characters without
     * disturbing the relative positions of the remaining characters. (ie, "ACE"
     * is a subsequence of "ABCDE" while "AEC" is not).
     *
     * @param S,
     *            T: Two string.
     * @return: Count the number of distinct subsequences
     */
    @tags.String
    @tags.DynamicProgramming
    public int numDistinct(String S, String T) {
        // TODO: get used to this implicit recurrence relation
        // Take away from this problem is draw the matrix first if the only
        // unclear thing is the recurrence relation.
        if (S == null || T == null) {
            return 0;
        }

        int[][] nums = new int[S.length() + 1][T.length() + 1];

        for (int i = 0; i <= S.length(); i++) {
            nums[i][0] = 1;
        }
        for (int i = 1; i <= S.length(); i++) {
            for (int j = 1; j <= T.length(); j++) {
                nums[i][j] = nums[i - 1][j];
                if (S.charAt(i - 1) == T.charAt(j - 1)) {
                    nums[i][j] += nums[i - 1][j - 1];
                }
            }
        }
        return nums[S.length()][T.length()];
    }

    /**
     * Word Break
     *
     * Given a string s and a dictionary of words dict, determine if s can be
     * break into a space-separated sequence of one or more dictionary words.
     *
     * For example, given s = "leetcode", dict = ["leet", "code"]. Return true
     * because "leetcode" can be segmented as "leet code".
     *
     * Bottom up DP seems quick useful
     *
     * @param s:
     *            A string s
     * @param dict:
     *            A dictionary of words dict
     */
    @tags.String
    @tags.DynamicProgramming
    public boolean wordBreak(String s, Set<String> dict) {
        if (s == null || dict == null)
            return false;

        // wb[i] = breakable before char at i
        boolean[] wb = new boolean[s.length() + 1];
        wb[0] = true;

        // find the longest word in dict
        int maxLen = 0;
        for (String word : dict) {
            maxLen = Math.max(maxLen, word.length());
        }

        for (int i = 1; i < wb.length; i++) {
            for (int j = i - 1; j >= 0 && i - j <= maxLen; j--) {
                if (wb[j] && dict.contains(s.substring(j, i))) {
                    wb[i] = true;
                    break;
                }
            }
        }

        return wb[wb.length - 1];
    }

    /**
     * Word Break
     *
     * Bottom up DP solution seems quick useful
     *
     * @param s:
     *            A string s
     * @param dict:
     *            A dictionary of words dict
     */
    @tags.String
    @tags.DynamicProgramming
    public boolean wordBreakBottomUp(String s, Set<String> dict) {
        if (s == null || dict == null)
            return false;

        int len = s.length();
        boolean[] dp = new boolean[len + 1];
        dp[len] = true;

        for (int i = len - 1; i >= 0; i--) {
            for (int j = i; j < len; j++) {
                String str = s.substring(i, j + 1);
                if (dict.contains(str) && dp[j + 1]) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[0];
    }

    /**
     * Palindrome Partitioning II
     *
     * Given a string s, partition s such that every substring of the partition
     * is a palindrome.
     *
     * Return the minimum cuts needed for a palindrome partitioning of s.
     *
     * For example, given s = "aab", Return 1 since the palindrome partitioning
     * ["aa","b"] could be produced using 1 cut.
     */
    public int minCut(String s) {
        // TODO get used to defining dp array in reversed order
        int len = s.length();
        boolean[][] isPal = new boolean[len][len];
        int[] dp = new int[len + 1];

        for (int i = 0; i <= len; i++) {
            dp[i] = len - 1 - i;
        }

        for (int i = len - 2; i >= 0; i--) {
            for (int j = i; j < len; j++) {
                if (s.charAt(i) == s.charAt(j) && (j <= i + 2 || isPal[i + 1][j - 1])) {
                    isPal[i][j] = true;
                    dp[i] = Math.min(dp[i], dp[j + 1] + 1);
                }
            }
        }
        return dp[0];
    }

    /**
     * Longest Common Substring.
     *
     * Given two strings, find the longest common substring. Return the length
     * of it.
     *
     * Notice: The characters in substring should occur continuously in original
     * string. This is different with subsequence.
     *
     * Example: Given A = "ABCD", B = "CBCE", return 2.
     *
     * @param A,
     *            B: Two string.
     * @return: the length of the longest common substring.
     */
    @tags.String
    @tags.Site.LintCode
    public int longestCommonSubstring(String A, String B) {
        int maxlen = 0;
        int alen = A.length();
        int blen = B.length();

        for (int i = 0; i < alen; ++i) {
            for (int j = 0; j < blen; ++j) {
                int len = 0;
                while (i + len < alen && j + len < blen && A.charAt(i + len) == B.charAt(j + len)) {
                    len++;
                }
                if (len > maxlen)
                    maxlen = len;
            }
        }

        return maxlen;
    }

    /**
     * Longest Common Substring.
     *
     * DP solution, O(m * n).
     *
     * @param A,
     *            B: Two string.
     * @return: the length of the longest common substring.
     */
    @tags.String
    @tags.DynamicProgramming
    @tags.Site.LintCode
    public int longestCommonSubstringDP(String A, String B) {
        if (A == null || A.length() == 0 || B == null || B.length() == 0) {
            return 0;
        }

        int m = A.length();
        int n = B.length();

        int max = 0;
        int[][] dp = new int[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (A.charAt(i) == B.charAt(j)) {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = dp[i - 1][j - 1] + 1;
                    }
                    if (max < dp[i][j]) {
                        max = dp[i][j];
                    }
                }
            }
        }

        return max;
    }

    /**
     * Longest Common Subsequence.
     *
     * Given two strings, find the longest common subsequence (LCS). Your code
     * should return the length of LCS.
     *
     * Example For "ABCD" and "EDCA", the LCS is "A" (or "D", "C"), return 1.
     * For "ABCD" and "EACB", the LCS is "AC", return 2.
     *
     * @param A,
     *            B: Two strings.
     * @return: The length of longest common subsequence of A and B.
     */
    @tags.DynamicProgramming
    @tags.Site.LintCode
    public int longestCommonSubsequence(String A, String B) {
        if (A == null || A.length() == 0 || B == null || B.length() == 0) {
            return 0;
        }

        int aLen = A.length(), bLen = B.length();
        int[][] dp = new int[aLen][bLen];

        for (int i = 0; i < aLen; i++) {
            for (int j = 0; j < bLen; j++) {
                if (A.charAt(i) == B.charAt(j)) {
                    dp[i][j] = 1;
                }

                if (i != 0 && j != 0) {
                    dp[i][j] += dp[i - 1][j - 1];
                }
                if (i != 0) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - 1][j]);
                }
                if (j != 0) {
                    dp[i][j] = Math.max(dp[i][j], dp[i][j - 1]);
                }
            }
        }

        return dp[aLen - 1][bLen - 1];
    }

    /**
     * Interleaving String.
     *
     * Given three strings: s1, s2, s3, determine whether s3 is formed by the
     * interleaving of s1 and s2.
     *
     * For s1 = "aabcc", s2 = "dbbca". When s3 = "aadbbcbcac", return true. When
     * s3 = "aadbbbaccc", return false.
     *
     * @param s1,
     *            s2, s3: As description.
     * @return: true or false.
     */
    @tags.DynamicProgramming
    public boolean isInterleave(String s1, String s2, String s3) {
        if (s1 == null || s2 == null || s3 == null) {
            throw new IllegalArgumentException("Input strings cannot be null.");
        }

        int m = s1.length(), n = s2.length();
        if (m + n != s3.length()) {
            return false;
        }

        // Backward dp population from right bottom corner
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[m][n] = true;
        for (int i = m - 1; i >= 0; i--) {
            if (s1.charAt(i) != s3.charAt(n + i)) {
                break;
            }
            dp[i][n] = true;
        }
        for (int j = n - 1; j >= 0; j--) {
            if (s2.charAt(j) != s3.charAt(m + j)) {
                break;
            }
            dp[m][j] = true;
        }

        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                if ((s1.charAt(i) == s3.charAt(i + j) && dp[i + 1][j])
                        || (s2.charAt(j) == s3.charAt(i + j) && dp[i][j + 1])) {
                    dp[i][j] = true;
                }
            }
        }

        return dp[0][0];
    }

    // ------------------------ Main Function and Tests ------------------------

    public void test() {
        int[] nums = { 5, 4, 1, 2, 3 };
        longestIncreasingSubsequence2(nums);
        String s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac";
        isInterleave(s1, s2, s3);
    }

    public static void main(String[] args) {
        DynamicProgramming dp = new DynamicProgramming();
        dp.test();
    }
}
