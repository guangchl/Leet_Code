package ninechapter;

import java.util.ArrayList;

public class DP {
	
	/**
	 * Climbing Stairs
	 * @param n
	 * @return
	 */
	public int climbStairs(int n) {
		// leetcode don't test this case, should ask interviewer
        if (n == 0) {
            return 0;
        }
        
        // just like fibonacci, but here return b instead
        int a = 0;
        int b = 1;
        
        while (n-- > 0) {
            int temp = a + b;
            a = b;
            b = temp;
        }
        
        return b;
    }
	
	/**
	 * Triangle
	 * 
	 * Given a triangle, find the minimum path sum from top to bottom. Each step
	 * you may move to adjacent numbers on the row below.
	 * 
	 * Space: O(n)
	 * Time: O(1)
	 */
	public int minimumTotal(ArrayList<ArrayList<Integer>> triangle) {
        if (triangle == null || triangle.size() == 0) {
            return 0;
        }
        
        int len = triangle.size();
        int[] dp = new int[len + 1];
        
        // start from the last level, calculate the minimum path from each node
        for (int i = len - 1; i >= 0; i--) {
            ArrayList<Integer> row = triangle.get(i);
            for (int j = 0; j < row.size(); j++) {
                dp[j] = Math.min(dp[j], dp[j + 1]) + row.get(j);
            }
        }
        
        return dp[0];
    }
	
	/**
	 * Unique Paths
	 */
	public int uniquePaths(int m, int n) {
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
	 * Additional obstacle matrix
	 */
	public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid == null || obstacleGrid.length == 0 || obstacleGrid[0].length == 0) {
            return 0;
        }
        
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int[] dp = new int[n];
        dp[0] = 1;

        for (int i = 0; i < m; i++) {
            dp[0] = (obstacleGrid[i][0] == 1 || dp[0] == 0) ? 0 : 1;
            for (int j = 1; j < n; j++) {
                dp[j] = (obstacleGrid[i][j] == 1) ? 0 : dp[j] + dp[j - 1];
            }
        }
        
        return dp[n - 1];
    }
	
	/**
	 * Jump Game
	 * 
	 * Given an array of non-negative integers, you are initially positioned at
	 * the first index of the array.
	 * 
	 * Each element in the array represents your maximum jump length at that
	 * position.
	 * 
	 * Determine if you are able to reach the last index.
	 */
	public boolean canJump(int[] A) {
        if (A == null || A.length < 2) {
            return true;
        }
        
        int dist = 0;
        
        for (int i = 0; i <= dist && i < A.length; i++) {
            if (dist < i + A[i]) {
                dist = i + A[i];
            }
        }
        
        return dist >= A.length - 1;
    }
	
	public DP() {
		// TODO Auto-generated constructor stub
	}

}
