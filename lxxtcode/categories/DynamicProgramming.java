package categories;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

import org.junit.Assert;
import org.junit.Test;

public class DynamicProgramming {

    // ---------------------------------------------------------------------- //
    // ------------------------------- MODELS ------------------------------- //
    // ---------------------------------------------------------------------- //

    /** Definition for binary tree */
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    // ---------------------------------------------------------------------- //
    // ------------------------------ PROBLEMS ------------------------------ //
    // ---------------------------------------------------------------------- //

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

    /** Unique Paths (1 dimensional Array). */
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
    @tags.Status.NeedPractice
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid == null || obstacleGrid.length == 0
                || obstacleGrid[0].length == 0) {
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
    @tags.Status.OK
    public int climbStairs(int n) {
        // Good question to ask interviewer about edge cases

        // just like fibonacci
        int a = 1, b = 1;
        while (n-- > 0) {
            int c = a + b;
            a = b;
            b = c;
        }
        return a;
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
    @tags.Status.NeedPractice
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
     * Jump Game.
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

        int n = A.length;
        int[] steps = new int[n];
        for (int i = 1; i < n; i++) {
            steps[i] = Integer.MAX_VALUE;
        }

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j <= i + A[i] && j < n; j++) {
                steps[j] = Math.min(steps[j], steps[i] + 1);
            }
        }

        return steps[n - 1] == Integer.MAX_VALUE ? -1 : steps[n - 1];
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
    @tags.Source.LeetCode
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
    @tags.Status.OK
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
     * Distinct Subsequences.
     *
     * Given a string S and a string T, count the number of distinct
     * subsequences of T in S.
     *
     * A subsequence of a string is a new string which is formed from the
     * original string by deleting some (can be none) of the characters without
     * disturbing the relative positions of the remaining characters. (ie, "ACE"
     * is a subsequence of "ABCDE" while "AEC" is not).
     *
     * Example: Given S = "rabbbit", T = "rabbit", return 3.
     *
     * Challenge: Do it in O(n2) time and O(n) memory. O(n2) memory is also
     * acceptable if you do not know how to optimize memory.
     *
     * @param S,
     *            T: Two string.
     * @return: Count the number of distinct subsequences
     */
    @tags.String
    @tags.DynamicProgramming
    @tags.Status.Hard
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
    @tags.Source.LintCode
    public int longestCommonSubstring(String A, String B) {
        int maxlen = 0;
        int alen = A.length();
        int blen = B.length();

        for (int i = 0; i < alen; ++i) {
            for (int j = 0; j < blen; ++j) {
                int len = 0;
                while (i + len < alen && j + len < blen
                        && A.charAt(i + len) == B.charAt(j + len)) {
                    len++;
                }
                if (len > maxlen)
                    maxlen = len;
            }
        }

        return maxlen;
    }

    /**
     * Longest Common Substring - DP solution, O(m * n).
     */
    @tags.String
    @tags.DynamicProgramming
    @tags.Source.LintCode
    @tags.Status.Easy
    public int longestCommonSubstringDP(String A, String B) {
        if (A == null || B == null) {
            return 0;
        }

        int m = A.length(), n = B.length();
        int[][] lcs = new int[m + 1][n + 1];
        int max = 0;

        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                if (A.charAt(i) == B.charAt(j)) {
                    lcs[i][j] = lcs[i + 1][j + 1] + 1;
                    max = Math.max(max, lcs[i][j]);
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
    @tags.Source.LintCode
    public int longestCommonSubsequence(String A, String B) {
        if (A == null || B == null || A.length() * B.length() == 0) {
            return 0;
        }

        int m = A.length(), n = B.length();
        int[][] lcs = new int[m + 1][n + 1];

        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                lcs[i][j] = Math.max(lcs[i + 1][j], lcs[i][j + 1]);
                if (A.charAt(i) == B.charAt(j)) {
                    lcs[i][j] = Math.max(lcs[i][j], lcs[i + 1][j + 1] + 1);
                }
            }
        }

        return lcs[0][0];
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
    @tags.Source.LintCode
    @tags.Status.OK
    public int longestIncreasingSubsequence(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }

        int[] lis = new int[nums.length];
        int max = 0;

        for (int i = 0; i < nums.length; i++) {
            lis[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    lis[i] = Math.max(lis[i], lis[j] + 1);
                }
            }
            max = Math.max(max, lis[i]);
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
    @tags.Status.OK
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

        for (int i = m; i >= 0; i--) {
            for (int j = n; j >= 0; j--) {
                if (i != m) {
                    dp[i][j] = s1.charAt(i) == s3.charAt(i + j) && dp[i + 1][j];
                }
                if (j != n && !dp[i][j]) {
                    dp[i][j] = s2.charAt(j) == s3.charAt(i + j) && dp[i][j + 1];
                }
            }
        }

        return dp[0][0];
    }

    /**
     * Decode Ways.
     *
     * A message containing letters from A-Z is being encoded to numbers using
     * the following mapping: 'A' -> 1 'B' -> 2 ... 'Z' -> 26. Given an encoded
     * message containing digits, determine the total number of ways to decode
     * it.
     *
     * @param s
     *            a string, encoded message
     * @return an integer, the number of ways decoding
     */
    @tags.String
    @tags.DynamicProgramming
    @tags.Company.Facebook
    @tags.Company.Microsoft
    @tags.Company.Uber
    public int numDecodings(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }

        int n = s.length();
        int[] ways = new int[n + 1];
        ways[n] = 1;
        ways[n - 1] = s.charAt(n - 1) != '0' ? 1 : 0;

        for (int i = n - 2; i >= 0; i--) {
            if (s.charAt(i) != '0') {
                // single digit
                ways[i] = ways[i + 1];

                // double digit
                int num = Integer.parseInt(s.substring(i, i + 2));
                if (num >= 1 && num <= 26) {
                    ways[i] += ways[i + 2];
                }
            }
        }

        return ways[0];
    }

    /**
     * Minimum Adjustment Cost
     *
     * Given an integer array, adjust each integers so that the difference of
     * every adjacent integers are not greater than a given number target. If
     * the array before adjustment is A, the array after adjustment is B, you
     * should minimize the sum of |A[i]-B[i]|
     *
     * Notice: You can assume each number in the array is a positive integer and
     * not greater than 100.
     *
     * Example: Given [1,4,2,3] and target = 1, one of the solutions is
     * [2,3,2,3], the adjustment cost is 2 and it's minimal. Return 2.
     *
     * @param A:
     *            An integer array.
     * @param target:
     *            An integer.
     */
    @tags.DynamicProgramming
    @tags.Backpack
    @tags.Source.LintCode
    @tags.Status.Hard
    public int MinAdjustmentCost(ArrayList<Integer> A, int target) {
        if (A == null || target < 0) {
            return 0;
        }

        int n = A.size();
        int[][] costs = new int[n + 1][100 + 1];

        for (int i = n - 1; i >= 0; i--) {
            for (int j = 1; j <= 100; j++) {
                int lower = Math.max(1, j - target);
                int upper = Math.min(100, j + target);
                costs[i][j] = costs[i + 1][upper];
                for (int k = lower; k < upper; k++) {
                    costs[i][j] = Math.min(costs[i][j], costs[i + 1][k]);
                }
                costs[i][j] += Math.abs(A.get(i) - j);
            }
        }

        int min = costs[0][1];
        for (int i = 1; i <= 100; i++) {
            min = Math.min(min, costs[0][i]);
        }

        return min;
    }

    /**
     * Perfect Squares.
     *
     * Given a positive integer n, find the least number of perfect square
     * numbers (for example, 1, 4, 9, 16, ...) which sum to n.
     *
     * For example, given n = 12, return 3 because 12 = 4 + 4 + 4; given n = 13,
     * return 2 because 13 = 4 + 9.
     *
     * @param n a positive integer
     * @return an integer
     */
    @tags.DynamicProgramming
    @tags.BFS
    @tags.Math
    @tags.Company.Google
    @tags.Status.NeedPractice
    public int numSquares(int n) {
        int[] dp = new int[n + 1];

        for (int i = 1; i <= n; i++) {
            dp[i] = dp[i - 1] + 1;
            for (int j = 1; j * j <= i; j++) {
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
            }
        }

        return dp[n];
    }

    /** Perfect Squares - pure math solution. */
    public int numSquares2(int n) {
        // Write your code here
        while (n % 4 == 0)
            n /= 4;
        if (n % 8 == 7)
            return 4;
        for (int i = 0; i * i <= n; ++i) {
            int j = (int) Math.sqrt(n * 1.0 - i * i);
            if (i * i + j * j == n) {
                int res = 0;
                if (i > 0)
                    res += 1;
                if (j > 0)
                    res += 1;
                return res;
            }
        }
        return 3;
    }

    /**
     * Dices Sum.
     *
     * Throw n dices, the sum of the dices' faces is S. Given n, find the all
     * possible value of S along with its probability.
     *
     * Example: Given n = 1, return [ [1, 0.17], [2, 0.17], [3, 0.17], [4,
     * 0.17], [5, 0.17], [6, 0.17]].
     *
     * Do the division at the end to avoid difference made by precision lost,
     * while use BigDecimal is another way.
     *
     * @param n
     *            an integer
     * @return a list of Map.Entry<sum, probability>
     */
    @tags.DynamicProgramming
    @tags.Math
    @tags.Probability
    public List<Map.Entry<Integer, Double>> dicesSum(int n) {
        // init
        double[] sumsCount = new double[6 * n];
        for (int i = 0; i < 6; i++) {
            sumsCount[i] = 1;
        }

        // roll dice n times
        for (int i = 1; i < n; i++) {
            for (int j = 6 * i - 1; j >= i - 1; j--) {
                for (int k = 1; k <= 6; k++) {
                    sumsCount[j + k] += sumsCount[j];
                }
                sumsCount[j] = 0;
            }
        }

        List<Map.Entry<Integer, Double>> result = new ArrayList<>();
        double total = Math.pow(6, n);
        for (int i = n - 1; i < sumsCount.length; i++) {
            result.add(
                    new AbstractMap.SimpleEntry<>(i + 1, sumsCount[i] / total));
        }
        return result;
    }

    // ---------------------------------------------------------------------- //
    // ----------------------------- Word Break ----------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Word Break.
     *
     * Given a string s and a dictionary of words dict, determine if s can be
     * break into a space-separated sequence of one or more dictionary words.
     *
     * For example, given s = "leetcode", dict = ["leet", "code"]. Return true
     * because "leetcode" can be segmented as "leet code".
     *
     * Bottom up DP seems quick useful.
     *
     * @param s:
     *            A string s
     * @param dict:
     *            A dictionary of words dict
     */
    @tags.String
    @tags.DynamicProgramming
    @tags.Company.Amazon
    @tags.Company.Bloomberg
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Company.PocketGems
    @tags.Company.Uber
    @tags.Company.Yahoo
    public boolean wordBreak(String s, Set<String> wordDict) {
        if (s == null || wordDict == null)
            return false;

        int len = s.length();
        boolean[] canBreak = new boolean[len + 1];
        canBreak[len] = true;

        // find the longest word in dict
        int maxLen = 0;
        for (String word : wordDict) {
            maxLen = Math.max(maxLen, word.length());
        }

        for (int i = len - 1; i >= 0; i--) {
            for (int j = i + 1; j <= len && j - i <= maxLen; j++) {
                if (canBreak[j] && wordDict.contains(s.substring(i, j))) {
                    canBreak[i] = true;
                    break;
                }
            }
        }

        return canBreak[0];
    }

    /** Word Break - DFS solution (TLE). */
    public boolean wordBreakDFS(String s, Set<String> wordDict) {
        return wordBreak(s, 0, wordDict);
    }

    private boolean wordBreak(String s, int pos, Set<String> wordDict) {
        if (pos == s.length()) {
            return true;
        }

        for (int i = pos; i < s.length(); i++) {
            String word = s.substring(pos, i + 1);
            if (wordDict.contains(word) && wordBreak(s, i + 1, wordDict)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Word Break II
     *
     * Given a string s and a dictionary of words dict, add spaces in s to
     * construct a sentence where each word is a valid dictionary word.
     *
     * Return all such possible sentences.
     *
     * For example, given s = "catsanddog", dict = ["cat", "cats", "and",
     * "sand", "dog"].
     *
     * A solution is ["cats and dog", "cat sand dog"].
     */
    @tags.DynamicProgramming
    @tags.Backtracking
    @tags.Company.Dropbox
    @tags.Company.Google
    @tags.Company.Snapchat
    @tags.Company.Twitter
    @tags.Company.Uber
    public List<String> wordBreakII(String s, Set<String> wordDict) {
        List<String> result = new ArrayList<>();
        if (s == null || s.length() == 0 || wordDict == null
                || wordDict.size() == 0) {
            return result;
        }

        // init dp list
        int len = s.length();
        List<List<Integer>> dp = new ArrayList<>();
        for (int i = 0; i <= len; i++) {
            dp.add(new ArrayList<Integer>());
        }

        // DP to find all possible link
        for (int i = len - 1; i >= 0; i--) {
            List<Integer> list = new ArrayList<>();
            for (int j = i + 1; j <= len; j++) {
                if ((j == len || !dp.get(j).isEmpty())
                        && wordDict.contains(s.substring(i, j))) {
                    list.add(j);
                }
            }
            dp.get(i).addAll(list);
        }

        // no result found
        if (dp.get(0).isEmpty()) {
            return result;
        }

        wordBreak(s, dp, 0, result, "");

        return result;
    }

    public void wordBreak(String s, List<List<Integer>> dp, int pos,
            List<String> result, String path) {
        if (pos == s.length()) {
            result.add(path.trim());
            return;
        }
        for (Integer next : dp.get(pos)) {
            wordBreak(s, dp, next, result, path + " " + s.substring(pos, next));
        }
    }

    // ---------------------------------------------------------------------- //
    // ------------------------------ Backpack ------------------------------ //
    // ---------------------------------------------------------------------- //

    /**
     * Backpack.
     *
     * Given n items with size Ai, an integer m denotes the size of a backpack.
     * How full you can fill this backpack?
     *
     * If we have 4 items with size [2, 3, 5, 7], the backpack size is 11, we
     * can select [2, 3, 5], so that the max size we can fill this backpack is
     * 10. If the backpack size is 12. we can select [2, 3, 7] so that we can
     * fulfill the backpack. You function should return the max size we can fill
     * in the given backpack.
     *
     * ±³°üÎÊÌâ¾Å½²[http://love-oriented.com/pack/], awesome work by Tianyi Cui.
     *
     * @param m:
     *            An integer m denotes the size of a backpack
     * @param A:
     *            Given n items with size A[i]
     * @return: The maximum size
     */
    @tags.DynamicProgramming
    @tags.Backpack
    @tags.Source.LintCode
    @tags.Status.Hard
    public int backPack(int m, int[] A) {
        if (m <= 0 || A == null || A.length == 0) {
            return 0;
        }

        // dp[i] = whether i can be reached
        boolean[] dp = new boolean[m + 1];
        dp[0] = true;

        for (Integer item : A) {
            for (int i = m; i >= item; i--) {
                if (!dp[i] && dp[i - item]) {
                    dp[i] = true;
                }
            }
        }

        for (int i = m; i > 0; i--) {
            if (dp[i]) {
                return i;
            }
        }
        return 0;
    }

    /**
     * Backpack II
     *
     * Given n items with size Ai and value Vi, and a backpack with size m.
     * What's the maximum value can you put into the backpack?
     *
     * Notice: You cannot divide item into small pieces and the total size of
     * items you choose should smaller or equal to m.
     *
     * Example: Given 4 items with size [2, 3, 5, 7] and value [1, 5, 2, 4], and
     * a backpack with size 10. The maximum value is 9.
     *
     * @param m:
     *            An integer m denotes the size of a backpack
     * @param A
     *            & V: Given n items with size A[i] and value V[i]
     * @return: The maximum value
     */
    @tags.DynamicProgramming
    @tags.Backpack
    @tags.Source.LintCode
    public int backPackII(int m, int[] A, int V[]) {
        if (m <= 0 || A == null || A.length == 0 || V == null
                || A.length != V.length) {
            return 0;
        }

        int[] dp = new int[m + 1];

        for (int i = 0; i < A.length; i++) {
            for (int j = m; j > 0; j--) {
                int pre = j - A[i];
                if (pre == 0 || (pre > 0 && dp[pre] > 0)) {
                    dp[j] = Math.max(dp[pre] + V[i], dp[j]);
                }
            }
        }

        int max = 0;
        for (int j = 1; j <= m; j++) {
            if (dp[j] > max) {
                max = dp[j];
            }
        }
        return max;
    }

    /**
     * Backpack III.
     *
     * Given n items with size Ai and value Vi, and a backpack with size m.
     * What's the maximum value can you put into the backpack?
     *
     * Notice: You cannot divide item into small pieces and the total size of
     * items you choose should smaller or equal to m.
     *
     * Example Given 4 items with size [2, 3, 5, 7] and value [1, 5, 2, 4], and
     * a backpack with size 10. The maximum value is 15.
     *
     * @param A
     *            an integer array
     * @param V
     *            an integer array
     * @param m
     *            an integer
     * @return an array
     */
    @tags.DynamicProgramming
    @tags.Backpack
    public int backPackIII(int[] A, int[] V, int m) {
        if (m <= 0 || A == null || A.length == 0 || V == null
                || V.length != A.length) {
            return 0;
        }

        int[] dp = new int[m + 1];
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j < A.length; j++) {
                int next = i + A[j];
                if (i == 0 || (next <= m && dp[i] > 0)) {
                    dp[next] = Math.max(dp[i] + V[j], dp[next]);
                }
            }
        }

        int max = 0;
        for (int i = 1; i <= m; i++) {
            if (dp[i] > max) {
                max = dp[i];
            }
        }
        return max;
    }

    // ---------------------------------------------------------------------- //
    // --------------- Wildcard / regular expression matching --------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Wildcard Matching.
     *
     * Implement wildcard pattern matching with support for '?' and '*'.
     *
     * '?' Matches any single character. '*' Matches any sequence of characters
     * (including the empty sequence).
     *
     * The matching should cover the entire input string (not partial).
     *
     * Some examples: isMatch("aa","a") ¡ú false, isMatch("aa","aa") ¡ú true,
     * isMatch("aaa","aa") ¡ú false, isMatch("aa", "*") ¡ú true, isMatch("aa",
     * "a*") ¡ú true, isMatch("ab", "?*") ¡ú true, isMatch("aab", "c*a*b") ¡ú false
     *
     * @param s:
     *            A string
     * @param p:
     *            A string includes "?" and "*"
     * @return: A boolean
     */
    @tags.Greedy
    @tags.String
    @tags.Backtracking
    @tags.DynamicProgramming
    @tags.Recursion
    @tags.DFS
    @tags.Company.Facebook
    @tags.Company.Google
    /** My DP solution */
    public boolean isMatch(String s, String p) {
        if (s == null || p == null) {
            return true;
        }

        int sLen = s.length();
        int pLen = p.length();
        boolean[][] match = new boolean[sLen + 1][pLen + 1];

        match[0][0] = true;
        for (int i = 1; i <= pLen; i++) {
            if (p.charAt(i - 1) == '*') {
                match[0][i] = true;
            } else {
                break;
            }
        }

        for (int i = 1; i <= sLen; i++) {
            for (int j = 1; j <= pLen; j++) {
                char pchar = p.charAt(j - 1);
                char schar = s.charAt(i - 1);
                if (pchar == '*') {
                    match[i][j] = match[i - 1][j] || match[i][j - 1];
                } else if (pchar == '?' || pchar == schar) {
                    match[i][j] = match[i - 1][j - 1];
                }
            }
        }

        return match[sLen][pLen];
    }

    /**
     * Wildcard Matching - DFS solution.
     *
     * This will exceed the time limit, the reason is multiple stars. In the
     * back tracking method below (isMatch3), once new star is found and
     * matched, the back track pointer will be moved forward, thus the program
     * won't go down the earlier back tracking branches.
     *
     * Multiple stars is bad not only when they are continuous, but also when
     * they are separated.
     */
    @tags.Greedy
    @tags.String
    @tags.Backtracking
    @tags.DynamicProgramming
    @tags.Recursion
    @tags.DFS
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Status.NeedPractice
    public boolean isMatch2(String s, String p) {
        if (s == null || p == null) {
            return true;
        }

        // optimization to merge continuous stars
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < p.length(); i++) {
            sb.append(p.charAt(i));
            if (p.charAt(i) == '*') {
                while (i + 1 < p.length() && p.charAt(i + 1) == '*') {
                    i++;
                }
            }
        }
        p = sb.toString();

        return isMatch(s, 0, p, 0);
    }

    private boolean isMatch(String s, int si, String p, int pi) {
        if (si == s.length()) {
            for (int i = pi; i < p.length(); i++) {
                if (p.charAt(i) != '*') {
                    return false;
                }
            }
            return true;
        } else if (pi == p.length()) {
            return false;
        }

        char schar = s.charAt(si);
        char pchar = p.charAt(pi);

        if (pchar == '*') {
            return isMatch(s, si + 1, p, pi) || isMatch(s, si, p, pi + 1);
        } else {
            if (pchar == '?' || pchar == schar) {
                return isMatch(s, si + 1, p, pi + 1);
            }
            return false;
        }
    }

    /**
     * Wildcard Matching.
     *
     * All time the best, hard to understand.
     */
    @tags.Greedy
    @tags.String
    @tags.Backtracking
    @tags.DynamicProgramming
    @tags.Recursion
    @tags.DFS
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Status.NeedPractice
    public boolean isMatch3(String s, String p) {
        if (s == null || p == null)
            return false;

        int is = 0, ip = 0; // search ptrs
        int ls = -1; // latest search ptr, next one to knock
        int lStar = -1; // latest star ptr

        while (true) {
            if (is == s.length()) {// end of s, check the rest of p
                for (int i = ip; i < p.length(); ++i) {
                    if (p.charAt(i) != '*')
                        return false;
                }
                return true;
            } else {
                if (ip < p.length()) {
                    if (s.charAt(is) == p.charAt(ip) || p.charAt(ip) == '?') {
                        // single match
                        is++;
                        ip++;
                        continue;
                    } else if (p.charAt(ip) == '*') {
                        // star, search next character in p
                        ls = is;
                        lStar = ip;
                        ip = lStar + 1;
                        continue;
                    }
                }

                // mismatch, check roll back
                if (ls >= 0) {
                    // roll back in the star position
                    ip = lStar + 1;
                    ls++;
                    is = ls;
                } else {// hard mismatch
                    return false;
                }
            }
        }
    }

    /**
     * Regular Expression Matching
     *
     * Implement regular expression matching with support for '.' and '*'.
     *
     * '.' Matches any single character. '*' Matches zero or more of the
     * preceding element.
     *
     * The matching should cover the entire input string (not partial).
     *
     * The function prototype should be: bool isMatch(const char *s, const char
     * *p)
     *
     * Example: isMatch("aa","a") ¡ú false isMatch("aa","aa") ¡ú true
     * isMatch("aaa","aa") ¡ú false isMatch("aa", "a*") ¡ú true isMatch("aa",
     * ".*") ¡ú true isMatch("ab", ".*") ¡ú true isMatch("aab", "c*a*b") ¡ú true
     *
     * @param s:
     *            A string
     * @param p:
     *            A string includes "." and "*"
     * @return: A boolean
     */
    @tags.String
    @tags.Backtracking
    @tags.DynamicProgramming
    @tags.Company.Facebook
    @tags.Company.Google
    /** DFS solution */
    public boolean isMatchRegular(String s, String p) {
        if (p.length() == 0) {
            return s.length() == 0;
        }

        if (p.length() == 1 || p.charAt(1) != '*') {
            if (s.length() == 0) {
                return false;
            }
            if (p.charAt(0) != '.' && p.charAt(0) != s.charAt(0)) {
                return false;
            } else {
                return isMatchRegular(s.substring(1), p.substring(1));
            }
        } else {
            if (isMatchRegular(s, p.substring(2))) {
                return true;
            } else {
                int i = 1;
                while (i <= s.length() && (p.charAt(0) == '.'
                        || s.charAt(i - 1) == p.charAt(0))) {
                    if (isMatchRegular(s.substring(i), p.substring(2))) {
                        return true;
                    }
                    i++;
                }
                return false;
            }
        }
    }

    /**
     * Regular Expression Matching.
     *
     * Non-recursive non-DP solution, copied logic from isMatch2.
     *
     * @param s
     * @param p
     * @return
     */
    @tags.String
    @tags.Backtracking
    @tags.DynamicProgramming
    @tags.Company.Facebook
    @tags.Company.Google
    /** iterative solution, awesome */
    public boolean isMatchRegular2(String s, String p) {
        if (s == null || p == null) {
            return false;
        }

        int is = 0, ip = 0;
        int ls = -1, lStar = -1;

        while (true) {
            if (is == s.length()) {
                for (int i = ip; i < p.length(); i += 2) {
                    if (p.charAt(i) != '*') {
                        return false;
                    }
                }
                return true;
            }

            if (ip < p.length()) {
                char s1 = s.charAt(is);
                char p1 = p.charAt(ip);
                char p2 = 'x';
                if (ip + 1 < p.length()) {
                    p2 = p.charAt(ip + 1);
                }

                if (p2 != '*') {
                    if (p1 == s1 || p1 == '.') {
                        ip++;
                        is++;
                        continue;
                    }
                } else {
                    if (p1 == '.') { // ".*" == "*" in wildcard matching
                        ls = is;
                        lStar = ip;
                        ip += 2;
                        continue;
                    } else { // repeat preceding letter
                        if (s1 == p1) {
                            int scnt = 0;
                            int pcnt = 0;
                            ip += 2;
                            while (ip < p.length() && p.charAt(ip) == s1) {
                                ip++;
                                pcnt++;
                            }
                            do {
                                is++;
                                scnt++;
                            } while (is < s.length() && s.charAt(is) == s1);
                            if (pcnt <= scnt) {
                                continue;
                            }
                        } else {
                            ip += 2;
                            continue;
                        }
                    }
                }
            }

            // hard no match, roll back to last *
            if (lStar != -1) {
                ls++;
                is = ls;
                ip = lStar + 2;
            } else {
                return false;
            }
        }
    }

    // ---------------------------------------------------------------------- //
    // ----------------------------- Palindrome ----------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Palindrome Partitioning.
     *
     * Given a string s, partition s such that every substring of the partition
     * is a palindrome. Return all possible palindrome partitioning of s.
     *
     * Example: Given s = "aab", return: [ ["aa","b"], ["a","a","b"] ].
     *
     * @param s:
     *            A string
     * @return: A list of lists of string
     */
    @tags.Backtracking
    @tags.DFS
    public List<List<String>> partition(String s) {
        int n = s.length();
        boolean[][] isPal = new boolean[n][n];
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (i == j) {
                    isPal[i][j] = true;
                } else if (j - i == 1) {
                    isPal[i][j] = s.charAt(i) == s.charAt(j);
                } else {
                    isPal[i][j] = isPal[i + 1][j - 1]
                            && s.charAt(i) == s.charAt(j);
                }
            }
        }

        List<List<String>> result = new ArrayList<>();
        List<String> path = new ArrayList<>();
        partition(isPal, s, path, 0, result);
        return result;
    }

    private void partition(boolean[][] isPal, String s, List<String> path,
            int pos, List<List<String>> result) {
        if (pos == s.length()) {
            result.add(new ArrayList<>(path));
        }
        for (int i = pos; i < s.length(); i++) {
            if (isPal[pos][i]) {
                path.add(s.substring(pos, i + 1));
                partition(isPal, s, path, i + 1, result);
                path.remove(path.size() - 1);
            }
        }
    }

    /**
     * Palindrome Partitioning II.
     *
     * Given a string s, cut s into some substrings such that every substring is
     * a palindrome. Return the minimum cuts needed for a palindrome
     * partitioning of s.
     *
     * Example: Given s = "aab", Return 1 since the palindrome partitioning
     * ["aa", "b"] could be produced using 1 cut.
     *
     * @param s
     *            a string
     * @return an integer
     */
    @tags.DynamicProgramming
    public int minCut(String s) {
        int n = s.length();
        int[] minCut = new int[n + 1];
        for (int i = 0; i < n; i++) {
            minCut[i] = n;
        }
        boolean[][] isPal = new boolean[n][n];

        for (int i = n - 1; i >= 0; i--) {
            for (int j = n - 1; j >= i; j--) {
                // find palindrome
                if (i == j) {
                    isPal[i][j] = true;
                } else if (j - i == 1) {
                    isPal[i][j] = s.charAt(i) == s.charAt(j);
                } else {
                    isPal[i][j] = isPal[i + 1][j - 1]
                            && s.charAt(i) == s.charAt(j);
                }

                // evaluate minCut
                if (isPal[i][j]) {
                    minCut[i] = Math.min(minCut[i], 1 + minCut[j + 1]);
                }
            }
        }

        return minCut[0] - 1;
    }

    /**
     * Palindrome Partitioning II - better solution.
     */
    @tags.DynamicProgramming
    public int minCut2(String s) {
        int len = s.length();
        boolean[][] isPal = new boolean[len][len];
        int[] dp = new int[len + 1];

        for (int i = 0; i <= len; i++)
            dp[i] = len - 1 - i;

        for (int i = len - 2; i >= 0; i--) {
            for (int j = i; j < len; j++) {
                if (s.charAt(i) == s.charAt(j)
                        && (j <= i + 2 || isPal[i + 1][j - 1])) {
                    isPal[i][j] = true;
                    dp[i] = Math.min(dp[i], dp[j + 1] + 1);
                }
            }
        }
        return dp[0];
    }

    /**
     * Longest Palindromic Substring.
     *
     * Given a string S, find the longest palindromic substring in S. You may
     * assume that the maximum length of S is 1000, and there exists one unique
     * longest palindromic substring.
     *
     * Example: Given the string = "abcdzdcab", return "cdzdc".
     *
     * Challenge: O(n2) time is acceptable. Can you do it in O(n) time.
     *
     * There is an O(n) solution with Manacher¡¯s Algorithm.
     *
     * @param s
     *            input string
     * @return the longest palindromic substring
     */
    @tags.String
    public String longestPalindrome(String s) {
        int maxLen = 0;
        String pal = "";
        for (int i = 0; i < s.length() - maxLen / 2; i++) {
            // left middle of even number characters
            int left = i, right = i + 1;
            int len = 0;
            while (left >= 0 && right < s.length()
                    && s.charAt(left) == s.charAt(right)) {
                len += 2;
                left--;
                right++;
            }
            if (len > maxLen) {
                maxLen = len;
                pal = s.substring(++left, right);
            }

            // middle of odd number characters
            left = i - 1;
            right = i + 1;
            len = 1;
            while (left >= 0 && right < s.length()
                    && s.charAt(left) == s.charAt(right)) {
                len += 2;
                left--;
                right++;
            }

            if (len > maxLen) {
                maxLen = len;
                pal = s.substring(++left, right);
            }
        }

        return pal;
    }

    /**
     * Shortest Palindrome.
     *
     * Given a string S, you are allowed to convert it to a palindrome by adding
     * characters in front of it. Find and return the shortest palindrome you
     * can find by performing this transformation.
     *
     * For example: Given "aacecaaa", return "aaacecaaa". Given "abcd", return
     * "dcbabcd".
     *
     * @param s
     * @return
     */
    @tags.String
    @tags.Company.Google
    @tags.Company.PocketGems
    @tags.Status.OK
    public String shortestPalindrome(String s) {
        if (s == null || s.length() < 2) {
            return s;
        }

        int n = s.length();
        int start = 0, end = n - 1;

        while (start < end) {
            if (isPalindrome(s, start, end)) {
                break;
            }
            end--;
        }

        return new StringBuilder().append(s.substring(end + 1)).reverse()
                .append(s).toString();
    }

    private boolean isPalindrome(String s, int start, int end) {
        while (start < end) {
            if (s.charAt(start++) != s.charAt(end--)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Palindrome Permutation.
     *
     * Given a string, determine if a permutation of the string could form a
     * palindrome.
     *
     * For example, "code" -> False, "aab" -> True, "carerac" -> True.
     *
     * Hint: Consider the palindromes of odd vs even length. What difference do
     * you notice?
     *
     * @param s
     * @return
     */
    @tags.HashTable
    @tags.Company.Bloomberg
    @tags.Company.Google
    @tags.Company.Uber
    public boolean canPermutePalindrome(String s) {
        if (s == null || s.length() < 2) {
            return true;
        }

        Set<Character> set = new HashSet<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (set.contains(c)) {
                set.remove(c);
            } else {
                set.add(c);
            }
        }

        return set.size() < 2;
    }

    /**
     * Nearest Palindrome.
     *
     * Find the closest palindrome number to the input.
     *
     * Example£º input 9, ouput 9. input 98, output 99. input 100, output 101.
     *
     * @param s
     * @return
     */
    @tags.String
    @tags.Company.Thumbtack
    @tags.Status.Hard
    public int nearestPalindrome(int n) {
        // TODO: this is not correct.
        if (n < 10) {
            return n;
        }

        StringBuilder num = new StringBuilder(String.valueOf(n));
        int start = 0, end = num.length() - 1;
        while (start < end) {
            num.setCharAt(end--, num.charAt(start++));
        }

        return Integer.parseInt(num.toString());
    }

    // ---------------------------------------------------------------------- //
    // --------------------- House Robber / Paint House --------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * House Robber.
     *
     * You are a professional robber planning to rob houses along a street. Each
     * house has a certain amount of money stashed, the only constraint stopping
     * you from robbing each of them is that adjacent houses have security
     * system connected and it will automatically contact the police if two
     * adjacent houses were broken into on the same night. Given a list of
     * non-negative integers representing the amount of money of each house,
     * determine the maximum amount of money you can rob tonight without
     * alerting the police.
     *
     * Example: Given [3, 8, 4], return 8.
     *
     * Challenge: O(n) time and O(1) memory.
     *
     * @param A:
     *            An array of non-negative integers. return: The maximum amount
     *            of money you can rob tonight
     */
    @tags.DynamicProgramming
    @tags.Company.Airbnb
    @tags.Company.LinkedIn
    @tags.Status.OK
    public long houseRobber(int[] A) {
        if (A == null || A.length == 0) {
            return 0;
        }

        int n = A.length;
        long[] max = new long[n + 1];
        max[n - 1] = A[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            max[i] = Math.max(max[i + 1], max[i + 2] + A[i]);
        }
        return max[0];
    }

    /**
     * House Robber II.
     *
     * After robbing those houses on that street, the thief has found himself a
     * new place for his thievery so that he will not get too much attention.
     * This time, all houses at this place are arranged in a circle. That means
     * the first house is the neighbor of the last one. Meanwhile, the security
     * system for these houses remain the same as for those in the previous
     * street. Given a list of non-negative integers representing the amount of
     * money of each house, determine the maximum amount of money you can rob
     * tonight without alerting the police.
     *
     * Notice: This is an extension of House Robber.
     *
     * Example: nums = [3,6,4], return 6.
     *
     * @param nums:
     *            An array of non-negative integers.
     * @return: The maximum amount of money you can rob tonight
     */
    @tags.DynamicProgramming
    @tags.Company.Microsoft
    @tags.Status.NeedPractice
    public int houseRobberII(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }

        // only one house
        int n = nums.length;
        if (n == 1) {
            return nums[0];
        }

        // forward order excluding first house
        int[] noFirst = new int[n];
        noFirst[1] = nums[1];
        for (int i = 2; i < n; i++) {
            noFirst[i] = Math.max(noFirst[i - 1], noFirst[i - 2] + nums[i]);
        }

        // backward order excluding last house
        int[] noLast = new int[n];
        noLast[n - 2] = nums[n - 2];
        for (int i = n - 3; i >= 0; i--) {
            noLast[i] = Math.max(noLast[i + 1], noLast[i + 2] + nums[i]);
        }

        return Math.max(noFirst[n - 1], noLast[0]);
    }

    /**
     * House Robber III.
     *
     * The thief has found himself a new place for his thievery again. There is
     * only one entrance to this area, called the "root." Besides the root, each
     * house has one and only one parent house. After a tour, the smart thief
     * realized that "all houses in this place forms a binary tree". It will
     * automatically contact the police if two directly-linked houses were
     * broken into on the same night.
     *
     * Determine the maximum amount of money the thief can rob tonight without
     * alerting the police.
     *
     * @param root:
     *            The root of binary tree.
     * @return: The maximum amount of money you can rob tonight
     */
    @tags.DFS
    @tags.Tree
    @tags.Company.Uber
    @tags.Status.Hard
    public int houseRobberIII(TreeNode root) {
        return Math.max(dp(root)[0], dp(root)[1]);
    }

    private int[] dp(TreeNode root) {
        // result[0] -> rob root, result[1] -> don't rob root
        int[] result = new int[2];
        if (root == null) {
            return result;
        }

        int[] left = dp(root.left);
        int[] right = dp(root.right);

        result[0] = root.val + left[1] + right[1];
        result[1] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);

        return result;
    }

    /**
     * Paint Fence.
     *
     * There is a fence with n posts, each post can be painted with one of the k
     * colors. You have to paint all the posts such that no more than two
     * adjacent fence posts have the same color. Return the total number of ways
     * you can paint the fence.
     *
     * Notice: n and k are non-negative integers.
     *
     * Example: Given n=3, k=2 return 6.
     *
     * @param n
     *            non-negative integer, n posts
     * @param k
     *            non-negative integer, k colors
     * @return an integer, the total number of ways
     */
    @tags.DynamicProgramming
    @tags.Company.Google
    public int numWays(int n, int k) {
        if (n <= 0 || k <= 0) {
            return 0;
        } else if (n == 1) { // easy to forget
            return k;
        }

        // init dp array
        // cannot count = new int[n + 1] and count[n] = 1
        int[] count = new int[n];
        count[n - 1] = k;
        count[n - 2] = k * k;

        for (int i = n - 3; i >= 0; i--) {
            // start with 2 different colors + start with 2 same colors
            count[i] = count[i + 1] * (k - 1) + count[i + 2] * (k - 1);
        }

        return count[0];
    }

    /**
     * Paint House.
     *
     * There are a row of n houses, each house can be painted with one of the
     * three colors: red, blue or green. The cost of painting each house with a
     * certain color is different. You have to paint all the houses such that no
     * two adjacent houses have the same color. The cost of painting each house
     * with a certain color is represented by a n x 3 cost matrix. For example,
     * costs[0][0] is the cost of painting house 0 with color red; costs[1][2]
     * is the cost of painting house 1 with color green, and so on... Find the
     * minimum cost to paint all houses.
     *
     * Notice: All costs are positive integers.
     *
     * Example: Given costs = [[14,2,11],[11,14,5],[14,3,10]], return 10. House
     * 0 is blue, house 1 is green, house 2 is blue, 2 + 5 + 3 = 10.
     *
     * @param costs
     *            n x 3 cost matrix
     * @return an integer, the minimum cost to paint all houses
     */
    @tags.DynamicProgramming
    @tags.Company.LinkedIn
    public int minCost(int[][] costs) {
        if (costs == null || costs.length == 0) {
            return 0;
        }
        int n = costs.length;
        int[][] dp = new int[n + 1][3];
        for (int i = n - 1; i >= 0; i--) {
            for (int j = 0; j < 3; j++) {
                dp[i][j] = costs[i][j] + Math.min(dp[i + 1][(j + 1) % 3],
                        dp[i + 1][(j + 2) % 3]);
            }
        }

        int min = dp[0][0];
        min = Math.min(min, dp[0][1]);
        min = Math.min(min, dp[0][2]);
        return min;
    }

    /**
     * Paint House II - O(nk).
     *
     * There are a row of n houses, each house can be painted with one of the k
     * colors. The cost of painting each house with a certain color is
     * different. You have to paint all the houses such that no two adjacent
     * houses have the same color. The cost of painting each house with a
     * certain color is represented by a n x k cost matrix. For example,
     * costs[0][0] is the cost of painting house 0 with color 0; costs[1][2] is
     * the cost of painting house 1 with color 2, and so on... Find the minimum
     * cost to paint all houses.
     *
     * Notice: All costs are positive integers.
     *
     * Example: Given n = 3, k = 3, costs = [[14,2,11],[11,14,5],[14,3,10]]
     * return 10. house 0 is color 2, house 1 is color 3, house 2 is color 2, 2
     * + 5 + 3 = 10.
     *
     * Challenge: Could you solve it in O(nk)?
     *
     * @param costs
     *            n x k cost matrix
     * @return an integer, the minimum cost to paint all houses
     */
    public int minCostII(int[][] costs) {
        if (costs == null || costs.length == 0 || costs[0].length == 0) {
            return 0;
        }

        int n = costs.length, k = costs[0].length;
        int[][] dp = new int[n][k];
        PriorityQueue<Integer> pq = new PriorityQueue<>(3,
                Collections.reverseOrder());

        // init
        for (int i = 0; i < k; i++) {
            dp[n - 1][i] = costs[n - 1][i];
            pq.offer(dp[n - 1][i]);
            if (pq.size() == 3) {
                pq.poll();
            }
        }

        // put min 2 numbers in min heap
        PriorityQueue<Integer> min = new PriorityQueue<>(2);
        while (!pq.isEmpty()) {
            min.offer(pq.poll());
        }

        // bottom up dp
        for (int i = n - 2; i >= 0; i--) {
            PriorityQueue<Integer> newMin = new PriorityQueue<>(3,
                    Collections.reverseOrder());
            for (int j = 0; j < k; j++) {
                if (dp[i + 1][j] == min.peek()) {
                    int tmp = min.poll();
                    dp[i][j] = costs[i][j] + min.peek();
                    min.offer(tmp);
                } else {
                    dp[i][j] = costs[i][j] + min.peek();
                }

                // add min for next round
                newMin.offer(dp[i][j]);
                if (newMin.size() == 3) {
                    newMin.poll();
                }
            }

            // put min 2 numbers in min heap
            min.clear();
            while (!newMin.isEmpty()) {
                min.offer(newMin.poll());
            }
        }

        return min.peek();
    }

    // ---------------------------------------------------------------------- //
    // ----------------------------- UNIT TESTS ----------------------------- //
    // ---------------------------------------------------------------------- //

    @Test
    public void tests() {
        int[] nums = { 5, 4, 1, 2, 3 };
        longestIncreasingSubsequence2(nums);

        String s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac";
        assert (isInterleave(s1, s2, s3));

        System.out.println(numDecodings("650"));

        String s = "abbabaaabbabbaababbabbbbbabbbabbbabaaaaababababbbabababaabbababaabbbbbbaaaabababbbaabbbbaabbbbababababbaabbaababaabbbababababbbbaaabbbbbabaaaabbababbbbaababaabbababbbbbababbbabaaaaaaaabbbbbaabaaababaaaabb",
                p = "**aa*****ba*a*bb**aa*ab****a*aaaaaa***a*aaaa**bbabb*b*b**aaaaaaaaa*a********ba*bbb***a*ba*bb*bb**a*b*bb";
        s = "bbbba";
        p = "?*a*a";
        p = "b*a";
        System.out.println("Wildcard matching: " + isMatch2(s, p));
        System.out.println("Regular expression: " + isMatchRegular2(s, p));

        partitionTests();

        wordBreakIITest();
    }

    private void partitionTests() {
        String s = "abbab";
        List<List<String>> expected = new ArrayList<>();
        expected.add(Arrays.asList("abba", "b"));
        expected.add(Arrays.asList("a", "b", "bab"));
        expected.add(Arrays.asList("a", "bb", "a", "b"));
        expected.add(Arrays.asList("a", "b", "b", "a", "b"));

        List<List<String>> result = partition(s);
        Collections.sort(result, new Comparator<List<String>>() {
            @Override
            public int compare(List<String> o1, List<String> o2) {
                return o1.size() - o2.size();
            }
        });

        Assert.assertEquals(expected, result);
    }

    private void wordBreakIITest() {
        String s = "catsanddog";
        Set<String> wordDict = new HashSet<>();
        wordDict.addAll(Arrays.asList("cat","cats","and","sand","dog"));

        Assert.assertEquals(2, wordBreakII(s, wordDict).size());
    }
}
