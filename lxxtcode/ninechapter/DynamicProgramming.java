package ninechapter;

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
     * Too simple, it's just like Fibonacci, we can even make it O(logn) or O(1).
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
     * @param triangle: a list of lists of integers.
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
     * nlogn solution.
     * TODO: I don't understand yet.
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

    public void test() {
        int[] nums = { 5, 4, 1, 2, 3 };
        longestIncreasingSubsequence2(nums);
    }

    public static void main(String[] args) {
        DynamicProgramming dp = new DynamicProgramming();
        dp.test();
    }
}
