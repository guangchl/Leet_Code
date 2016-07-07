package categories;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Essential skill testers.
 * @author Guangcheng Lu
 */
public class Basic {

	/**
	 * Implement strStr()
	 *
	 * Returns a pointer to the first occurrence of needle in haystack, or null
	 * if needle is not part of haystack.
	 */
    @tags.BasicImplementation
    @tags.String
	public String strStr(String src, String dest) {
		if (src == null || dest == null) {
			return null;
		}
		
		int i, j;
		for (i = 0; i <= src.length() - dest.length(); i++) {
			for (j = 0; j < dest.length(); j++) {
				if (src.charAt(i + j) != dest.charAt(j)) {
					break;
				}
			}
			
			if (j == dest.length()) {
				return src.substring(i, i + j);
			}
		}
		
		return null;
	}

	/** KMP */
    public String kmp(String haystack, String needle) {
        int m = haystack.length();
        int n = needle.length();
        if (n == 0) {
            return haystack;
        } else if (m < n) {
        	return null;
        }
        
        // construct cover of needle
        int[] cover = new int[n];
        int iter = 0;
        for (int i = 1; i < n; i++) {
            while (i < n && needle.charAt(i) == needle.charAt(iter)) {
                cover[i] = cover[i - 1] + 1;
                i++;
                iter++;
            }
            iter = 0;
        }

        int i = 0;
        int j = 0;
        while (i < m && j < n && m - i >= n - j) {
       		if (haystack.charAt(i) != needle.charAt(j)) {
       			if (j == 0) {
       				i += 1;
       			} else {
           			j = cover[j - 1];
       			}
       		} else {
       			i++;
       			j++;
       		}
        }
        
        return (j == n) ? haystack.substring(i - n) : null;
    }

    /**
     * Subsets.
     *
     * DFS solution.
     *
     * Given a set of distinct integers, S, return all possible subsets.
     *
     * Note: Elements in a subset must be in non-descending order. The solution
     * set must not contain duplicate subsets.
     */
    public ArrayList<ArrayList<Integer>> subsets(int[] S) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
        Arrays.sort(S); // not necessary
        ArrayList<Integer> tmp = new ArrayList<Integer>();
        res.add(tmp);
        dfsSubset(res, tmp, S, 0);
        return res;
    }

    private void dfsSubset(ArrayList<ArrayList<Integer>> res, ArrayList<Integer> tmp,
            int[] S, int pos) {
        for (int i = pos; i < S.length; i++) {
            tmp.add(S[i]);
            res.add(new ArrayList<Integer>(tmp));
            dfsSubset(res, tmp, S, i + 1);
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * Subsets.
     *
     * Iterative solution.
     *
     * @param S: A set of numbers.
     * @return: A list of lists. All valid subsets.
     */
    public ArrayList<ArrayList<Integer>> subsetsIter(int[] nums) {
        ArrayList<ArrayList<Integer>> subsets = new ArrayList<>();

        if (nums == null) {
            return subsets;
        }

        subsets.add(new ArrayList<Integer>());

        Arrays.sort(nums);

        for (Integer i : nums) {
            ArrayList<ArrayList<Integer>> newSets = new ArrayList<>();

            for (ArrayList<Integer> set : subsets) {
                ArrayList<Integer> newSet = new ArrayList<>(set);
                newSet.add(i);
                newSets.add(newSet);
            }

            subsets.addAll(newSets);
        }

        return subsets;
    }

    /**
     * 2. Subsets II
     *
     * Given a list of numbers that may has duplicate numbers, return all
     * possible subsets.
     *
     * Notice: Each element in a subset must be in non-descending order. The
     * ordering between two subsets is free. The solution set must not contain
     * duplicate subsets.
     */
    public ArrayList<ArrayList<Integer>> subsetsWithDup(int[] num) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
        Arrays.sort(num);
        ArrayList<Integer> tmp = new ArrayList<Integer>();
        res.add(tmp);
        dfsSubsetWithDup(res, tmp, num, 0);
        return res;
    }

    private void dfsSubsetWithDup(ArrayList<ArrayList<Integer>> res,
            ArrayList<Integer> tmp, int[] num, int pos) {
        for (int i = pos; i < num.length; i++) {
            tmp.add(num[i]);
            res.add(new ArrayList<Integer>(tmp));
            dfsSubsetWithDup(res, tmp, num, i + 1);
            tmp.remove(tmp.size() - 1);
            // the only one line difference
            while (i < num.length - 1 && num[i] == num[i + 1]) i++;
        }
    }

    /**
     * Subsets II.
     *
     * Iterative solution.
     *
     * @param S:
     *            A set of numbers.
     * @return: A list of lists. All valid subsets.
     */
    public ArrayList<ArrayList<Integer>> subsetsWithDupIterative(ArrayList<Integer> S) {
        if (S == null) {
            return new ArrayList<>();
        }

        // Sort the list
        Collections.sort(S);

        ArrayList<ArrayList<Integer>> subsets = new ArrayList<>();
        subsets.add(new ArrayList<Integer>());

        for (int i = 0; i < S.size(); i++) {
            int num = S.get(i), count = 1;
            while (i + 1 < S.size() && S.get(i + 1) == num) {
                i++;
                count++;
            }

            ArrayList<ArrayList<Integer>> newSets = new ArrayList<>();
            for (ArrayList<Integer> oldSet : subsets) {
                int countDown = count;
                while (countDown-- > 0) {
                    ArrayList<Integer> newSet = new ArrayList<>(oldSet);
                    newSet.add(num);
                    newSets.add(newSet);
                    oldSet = new ArrayList<>(newSet);
                }
            }
            subsets.addAll(newSets);
        }

        return subsets;
    }

    /**
     * 3. Permutations
     *
     * Given a list of numbers, return all possible permutations.
     */
    public ArrayList<ArrayList<Integer>> permute(ArrayList<Integer> nums) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (nums == null) {
            return result;
        }

        ArrayList<Integer> temp = new ArrayList<>();
        dfsPermute(result, temp, nums, 0);

        return result;
    }

    private void dfsPermute(ArrayList<ArrayList<Integer>> result, ArrayList<Integer> temp,
             ArrayList<Integer> nums, int index) {
        if (index == nums.size()) {
            result.add(new ArrayList<>(temp));
            return;
        }

        for (int i = 0; i <= temp.size(); i++) {
            temp.add(i, nums.get(index));
            dfsPermute(result, temp, nums, index + 1);
            temp.remove(i);
        }
    }

    /**
     * Permutation.
     *
     * My iterative solution.
     */
    public ArrayList<ArrayList<Integer>> permute(int[] num) {
        ArrayList<ArrayList<Integer>> permutations = new ArrayList<ArrayList<Integer>>();

        if (num.length == 0) {
            return permutations;
        }

        // add a initial empty list
        permutations.add(new ArrayList<Integer>());

        // add one integer in original array each time
        for (Integer i : num) {
            // construct a new list to new generated permutations
            ArrayList<ArrayList<Integer>> update = new ArrayList<ArrayList<Integer>>();

            // add the integer to every old permutation
            for (ArrayList<Integer> permutation : permutations) {
                // add the new integer to any possible position
                for (int j = 0; j < permutation.size() + 1; j++) {
                    ArrayList<Integer> newPermutation = new ArrayList<Integer>();
                    newPermutation.addAll(permutation); // add existing elements
                    newPermutation.add(j, i); // add new integer at position j
                    update.add(newPermutation);
                }
            }

            // set the result to updated list of permutations
            permutations = update;
        }

        return permutations;
    }

    /**
     * 4. Permutation II.
     *
     * Given a list of numbers with duplicate number in it. Find all unique
     * permutations.
     */
    public ArrayList<ArrayList<Integer>> permuteUnique(int[] num) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
        if (num == null || num.length == 0) {
            return result;
        }

        ArrayList<Integer> list = new ArrayList<Integer>();
        int[] visited = new int[num.length];
        Arrays.sort(num);
        permuteUniqueHelper(result, list, visited, num);

        return result;
    }

    private void permuteUniqueHelper(ArrayList<ArrayList<Integer>> result,
            ArrayList<Integer> list, int[] visited, int[] num) {
        if (list.size() == num.length) {
            result.add(new ArrayList<Integer>(list));
            return;
        }

        for (int i = 0; i < num.length; i++) {
            if (visited[i] == 1
                    || (i != 0 && num[i] == num[i - 1] && visited[i - 1] == 0)) {
                continue;
            }

            visited[i] = 1;
            list.add(num[i]);
            permuteUniqueHelper(result, list, visited, num);
            list.remove(list.size() - 1);
            visited[i] = 0;
        }
    }


    /**
     * Permutation II
     *
     * My dfs solution.
     *
     * @param nums: A list of integers.
     * @return: A list of unique permutations.
     */
    public ArrayList<ArrayList<Integer>> permuteUnique(ArrayList<Integer> nums) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (nums == null) {
            return result;
        }

        ArrayList<Integer> temp = new ArrayList<>();
        dfs(result, temp, nums, 0);

        return result;
    }

    private void dfs(ArrayList<ArrayList<Integer>> result,
                     ArrayList<Integer> temp, ArrayList<Integer> nums,
                     int pos) {
        if (pos == nums.size()) {
            result.add(new ArrayList<>(temp));
            return;
        }

        int from = temp.lastIndexOf(nums.get(pos)) + 1;
        for (int i = from; i <= temp.size(); i++) {
            temp.add(i, nums.get(pos));
            dfs(result, temp, nums, pos + 1);
            temp.remove(i);
        }
    }

    /**
     * Permutations II
     *
     * My iterative solution.
     */
    public ArrayList<ArrayList<Integer>> permuteUniqueIterative(int[] num) {
        ArrayList<ArrayList<Integer>> permutations = new ArrayList<ArrayList<Integer>>();

        if (num.length == 0) {
            return permutations;
        }

        permutations.add(new ArrayList<Integer>());

        // add one number to the permutations at one time
        for (Integer i : num) {
            ArrayList<ArrayList<Integer>> update = new ArrayList<ArrayList<Integer>>();

            for (ArrayList<Integer> permutation : permutations) {
                int from = permutation.lastIndexOf(i) + 1;
                for (int j = from; j < permutation.size() + 1; j++) {
                    ArrayList<Integer> newPermutation = new ArrayList<Integer>();
                    newPermutation.addAll(permutation);
                    newPermutation.add(j, i);
                    update.add(newPermutation);
                }
            }

            permutations = update;
        }

        return permutations;
    }

    // --------------------------- OLD ---------------------------

    /**
	 * Reverse Words in a String
	 */
	public String reverseWords(String s) {
        if (s == null || s.length() == 0) {
            return "";
        }
        
        int index = s.length() - 1;
        StringBuffer sb = new StringBuffer();
        
        while (index >= 0) {
            int end = index;
            while (end >= 0 && s.charAt(end) == ' ') {
                end--;
            }
            if (end < 0) {
                break;
            }
            
            int start = end;
            while (start >= 0 && s.charAt(start) != ' ') {
                start--;
            }

            sb.append(s.substring(start + 1, end + 1));
            sb.append(' ');
            index = start - 1;
        }
        
        return (sb.length() > 0) ? sb.substring(0, sb.length() - 1) : "";
    }
	
	/**
	 * Single Number
	 * 
	 * Given an array of integers, every element appears twice except for one.
	 * Find that single one. Time: O(n). Space: O(0).
	 * 
	 * If there's no space constraint, Map should be a common solution
	 */
	public int singleNumber(int[] A) {
        if(A == null || A.length == 0) {
            return -1;
        }
        
		// Since A^B^A == B, xor every other element with first one of the A
		for (int i = 1; i < A.length; i++) {
			A[0] ^= A[i];
		}
		return A[0];
	}

	/**
	 * Single Number II
	 * 
	 * Given an array of integers, every element appears three times except for
	 * one. Find that single one. Time: O(n). Space: O(0).
	 * 
	 * So tricky!!! Three bitmask variables.
	 */
	public int singleNumber2(int[] A) {
		int ones = 0; // represent the ith bit has appear once
		int twos = 0; // represent the ith bit has appear twice
		int threes = 0; // represent the ith bit has appear three times

		for (int i = 0; i < A.length; i++) {
			threes = (threes & ~A[i]) | (twos & A[i]);
			twos = (twos & ~A[i]) | (ones & A[i]);
			ones = (ones ^ A[i]) & ~(threes | twos);
		}

		return ones;
		// Another solution
		// int ones = 0, twos = 0, threes = 0;
		// for (int i = 0; i < n; i++) {
		// twos |= ones & A[i];
		// ones ^= A[i];
		// threes = ones & twos;
		// ones &= ~threes;
		// twos &= ~threes;
		// }
		// return ones;
	}
	
	/** Another approach */
	public int singleNumber22(int[] A) {
        if (A == null || A.length == 0) {
            return -1;
        }
        int result=0;
        int[] bits = new int[32];
        for (int i = 0; i < 32; i++) {
            for(int j = 0; j < A.length; j++) {
                bits[i] += A[j] >> i & 1;
                bits[i] %= 3;
            }

            result |= (bits[i] << i);
        }
        return result;
    }
	
	/**
	 * Best Time to Buy and Sell Stock
	 * 
	 * Say you have an array for which the ith element is the price of a given
	 * stock on day i. If you were only permitted to complete at most one
	 * transaction (ie, buy one and sell one share of the stock), design an
	 * algorithm to find the maximum profit.
	 */
	public int maxProfit(int[] prices) {
		if (prices == null || prices.length == 0) {
			return 0;
		}

		int min = prices[0];
		int profit = 0;

		for (int i = 1; i < prices.length; i++) {
			if (prices[i] > min) {
				profit = Math.max(profit, prices[i] - min);
			} else if (prices[i] < min) {
				min = prices[i];
			}
		}

		return profit;
	}

	/**
	 * Best Time to Buy and Sell Stock II
	 * 
	 * Say you have an array for which the ith element is the price of a given
	 * stock on day i. Design an algorithm to find the maximum profit. You may
	 * complete as many transactions as you like (ie, buy one and sell one share
	 * of the stock multiple times). However, you may not engage in multiple
	 * transactions at the same time (ie, you must sell the stock before you buy
	 * again).
	 */
	public int maxProfit2(int[] prices) {
		int profit = 0;

		for (int i = 1; i < prices.length; i++) {
			if (prices[i - 1] < prices[i]) {
				profit += prices[i] - prices[i - 1];
			}
		}

		return profit;
	}

	/**
	 * Best Time to Buy and Sell Stock III
	 * 
	 * Say you have an array for which the ith element is the price of a given
	 * stock on day i. Design an algorithm to find the maximum profit. You may
	 * complete at most two transactions.
	 * 
	 * Note: You may not engage in multiple transactions at the same time (ie,
	 * you must sell the stock before you buy again).
	 */
	public int maxProfit3(int[] prices) {
		int len = prices.length;
        if (len == 0)
            return 0;
        
        int[] forward = new int[len];
        int[] backward = new int[len];
        
        int min = prices[0];
        for (int i = 1; i < len; i++) {
            if (prices[i] > min) {
                forward[i] = Math.max(forward[i - 1], prices[i] - min);
            } else {
                if (prices[i] < min) {
                    min = prices[i];
                }
                forward[i] = forward[i - 1];
            }
        }
        
        int max = prices[len - 1];
        for (int i = len - 2; i >= 0; i--) {
            if (prices[i] < max)
                backward[i] = Math.max(backward[i + 1], max - prices[i]);
            else {
                if (prices[i] > max)
                    max = prices[i];
                backward[i] = backward[i + 1];
            }
        }
        
        int profit = 0;
        for (int i = 0; i < len; i++)
            profit = Math.max(profit, forward[i] + backward[i]);

        return profit;
	}
	
	/**
	 * Sort Colors
	 * 
	 * Given an array with n objects colored red, white or blue, sort them so
	 * that objects of the same color are adjacent, with the colors in the order
	 * red, white and blue.
	 * 
	 * Here, we will use the integers 0, 1, and 2 to represent the color red,
	 * white, and blue respectively.
	 * 
	 * Note: You are not suppose to use the library's sort function for this
	 * problem.
	 */
	public void sortColors(int[] A) {
		int start = 0; // index to put 0
        int end = A.length - 1; // index to put 2
        
        // traverse the array, move all 0 to beginning, all 1 to end
        for (int i = 0; i <= end;) {
            if (A[i] == 0) {
                A[i] = A[start];
                A[start] = 0;
                start++;
                i++;
            } else if (A[i] == 2) {
                A[i] = A[end];
                A[end] = 2;
                end--;
            } else {
                i++;
            }
        }
	}
	
	/**
	 * Sqrt(x)
	 * 
	 * Implement int sqrt(int x).
	 * 
	 * Compute and return the square root of x.
	 */
    public int sqrt(int x) {
		double error = 0.0000001f;
		double high = x;
		double low = 0;
		while (high - low > error) {
			double mid = (high + low) / 2;
			if (mid * mid > x) {
				high = mid;
			} else {
				low = mid;
			}
		}
		return (int) Math.floor(high);
    }
    
    /**
     * Pow(x, n)
     * 
     * Implement pow(x, n).
     */
    public double pow(double x, int n) {
        if (n == 0) {
            return 1;
        } else if (n == 1) {
            return x;
        } else if (n == -1) {
            return 1 / x;
        }
        
        double u = pow(x, n / 2);
        double result = u * u;
        
        if (n % 2 == 1) {
            result *= x;
        } else if (n % 2 == -1) {
            result /= x;
        }
        
        return result;
    }
    
    /**
	 * Divide Two Integers
	 * 
	 * Divide two integers without using multiplication, division and mod
	 * operator.
	 */
    public int divide(int dividend, int divisor) {
        boolean negative = dividend < 0 ^ divisor < 0;

        long a = Math.abs((long)dividend);
        long b = Math.abs((long)divisor);
        int ans = 0;

        while (a >= b) {
            int shift = 0;
            while ((b << shift) <= a) {
                shift++;
            }
            ans += 1 << (shift-1);
            a = a - (b << (shift-1));
        }

        return negative ? -ans : ans;
    }
    
    /**
	 * Valid Number
	 * 
	 * Validate if a given string is numeric.
	 * 
	 * Some examples: 
	 * "0" => true 
	 * " 0.1 " => true 
	 * "abc" => false 
	 * "1 a" => false
	 * "2e10" => true
	 * 
	 * Note: It is intended for the problem statement to be
	 * ambiguous. You should gather all requirements up front before
	 * implementing one.
	 */
	public boolean isNumberRegex(String s) {
		return s.matches("^\\s*[+-]?(\\d+|\\d*\\.\\d+|\\d+\\.\\d*)([eE][+-]?\\d+)?\\s*$");
	}

	public boolean isNumber(String s) {
		s = s.trim();
		if (s.length() > 0 && s.charAt(s.length() - 1) == 'e')
			return false; // avoid "3e" which is false
		String[] t = s.split("e");
		if (t.length == 0 || t.length > 2)
			return false;
		boolean res = valid(t[0], false);
		if (t.length > 1)
			res = res && valid(t[1], true);
		return res;
	}

	private boolean valid(String s, boolean hasDot) {
		if (s.length() > 0 && (s.charAt(0) == '+' || s.charAt(0) == '-')) // avoid
																			// "1+",
																			// "+",
																			// "+."
		s = s.substring(1);
		char[] arr = s.toCharArray();
		if (arr.length == 0 || s.equals("."))
			return false;
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] == '.') {
				if (hasDot)
					return false;
				hasDot = true;
			} else if (!('0' <= arr[i] && arr[i] <= '9')) {
				return false;
			}
		}
		return true;
	}
	
	/**
	 * Unique Paths
	 * 
	 * A robot is located at the top-left corner of a m x n grid (marked 'Start'
	 * in the diagram below).
	 * 
	 * The robot can only move either down or right at any point in time. The
	 * robot is trying to reach the bottom-right corner of the grid (marked
	 * 'Finish' in the diagram below).
	 * 
	 * How many possible unique paths are there?
	 * 
	 * Note: m and n will be at most 100.
	 */
	public int uniquePaths(int m, int n) {
		if (m == 0 || n == 0) {
			return 0;
		}
		
		int[][] pathNum = new int[m][n];
		
		// initialize the first line
		for (int i = 0; i < n; i++) {
			pathNum[0][i] = 1;
		}
		
		// initialize the first column
		for (int i = 1; i < m; i++) {
		    pathNum[i][0] = 1;
		}
		
		// fill all blanks left
		for (int i = 1; i < m; i++) {
		    for (int j = 1; j < n; j++) {
		        pathNum[i][j] = pathNum[i-1][j] + pathNum[i][j-1];
		    }
		}
		
		return pathNum[m-1][n-1];
	}
	
	/**
	 * Unique Paths II
	 * 
	 * Follow up for "Unique Paths":
	 * 
	 * Now consider if some obstacles are added to the grids. How many unique
	 * paths would there be?
	 * 
	 * An obstacle and empty space is marked as 1 and 0 respectively in the
	 * grid.
	 * 
	 * Note: You can only move either down or right at any point in time.
	 */
	public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        if (m == 0) {
            return 0;
        }
        
        int n = obstacleGrid[0].length;
        if (n == 0) {
            return 0;
        }
        
        // construct the cache matrix
        int[][] pathNum = new int[m][n];
        
        // fill the first column
        pathNum[0][0] = (obstacleGrid[0][0] == 0 ? 1 : 0);
        for (int i = 1; i < m; i++) {
            if (pathNum[i - 1][0] == 0 || obstacleGrid[i][0] == 1) {
                pathNum[i][0] = 0;
            } else {
                pathNum[i][0] = 1;
            }
        }
        
        // fill the first row
        for (int i = 1; i < n; i++) {
            if (pathNum[0][i - 1] == 0 || obstacleGrid[0][i] == 1) {
                pathNum[0][i] = 0;
            } else {
                pathNum[0][i] = 1;
            }
        }
        
        // fill all the remaining
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 1) {
                    pathNum[i][j] = 0;
                } else {
                    pathNum[i][j] = pathNum[i - 1][j] + pathNum[i][j - 1];
                }
            }
        }
        
        return pathNum[m - 1][n - 1];
    }
	
	/**
	 * Maximum Subarray
	 * 
	 * Find the contiguous subarray within an array (containing at least one
	 * number) which has the largest sum.
	 * 
	 * For example, given the array [-2,1,-3,4,-1,2,1,-5,4], the contiguous
	 * subarray [4,-1,2,1] has the largest sum = 6.
	 * 
	 * If you have figured out the O(n) solution, try coding another solution
	 * using the divide and conquer approach, which is more subtle.
	 */
	public int maxSubArray(int[] A) {
		int max = A[0];
		int endingMax = A[0];

		for (int i = 1; i < A.length; i++) {
			// calculate the possible max value ends at i
			endingMax = Math.max(A[i], endingMax + A[i]);

			// compare the max with the new possible max ends at i
			max = Math.max(endingMax, max);
		}

		return max;
	}

	/**
	 * N-Queens
	 * 
	 * Given an integer n, return all distinct solutions to the n-queens puzzle.
	 */
	public ArrayList<String[]> solveNQueens(int n) {
        // IMPORTANT: Please reset any member data you declared, as
        // the same Solution instance will be reused for each test case.
        ArrayList<String[]> result = new ArrayList<String[]>();
        int[] loc = new int[n];
        dfs(result, loc, 0, n);
        return result;
    }  
    
    public void dfs(ArrayList<String[]> result, int[] loc, int cur, int n){  
        if(cur == n) {
            printboard(result, loc, n);  
        } else{  
            for(int i = 0; i < n; i++){  
                loc[cur] = i;  
                if(isValid(loc, cur)) {
                    dfs(result, loc, cur + 1,n);
                }
            }  
        }  
    }  
      
    public boolean isValid(int[] loc, int cur){  
        for(int i = 0; i < cur; i++){  
            if(loc[i] == loc[cur] || Math.abs(loc[i] - loc[cur]) == (cur - i))  
                return false;  
        }  
        return true;  
    }  
          
    public void printboard(ArrayList<String[]> res, int[] loc, int n){  
        String[] ans = new String[n];  
        for(int i = 0; i < n; i++){  
            String row = new String();  
            for(int j = 0; j < n; j++){
                if(j == loc[i]) 
                    row += "Q";  
                else row += ".";  
            }  
            ans[i] = row;  
        }  
        res.add(ans);
    }
    
    /**
	 * N-Queens II
	 * 
	 * Follow up for N-Queens problem.
	 * 
	 * Now, instead outputting board configurations, return the total number of
	 * distinct solutions.
	 */
    public int totalNQueens(int n) {
        int[] locs = new int[n];
        return dfs(locs, 0, n);
    }  
    
    public int dfs(int[] locs, int cur, int n){
        int result = 0;
        
        if(cur == n)
            return 1;
        else{
            for(int i = 0; i < n; i++){
                locs[cur] = i;
                if(isValid(locs, cur))
                    result += dfs(locs, cur + 1, n);
            }
        }
        
        return result;
    }
    
    /**
	 * Anagrams
	 * 
	 * Given an array of strings, return all groups of strings that are
	 * anagrams.
	 * 
	 * Note: All inputs will be in lower-case.
	 */
    public ArrayList<String> anagrams(String[] strs) {
        ArrayList<String> strings = new ArrayList<String>();
        Map<String, ArrayList<String>> map = new HashMap<String, ArrayList<String>>();
        
        for (String s : strs) {
            char[] cs = s.toCharArray();
            Arrays.sort(cs);
            String ss = new String(cs);
            if (map.containsKey(ss)) {
                map.get(ss).add(s);
            } else {
                ArrayList<String> list = new ArrayList<String>();
                list.add(s);
                map.put(ss, list);
            }
        }
        
        for (ArrayList<String> value : map.values()) {
            if (value.size() > 1) {
                strings.addAll(value);
            }
        }
        
        return strings;
    }
    
    /**
	 * Two Sum
	 * 
	 * Given an array of integers, find two numbers such that they add up to a
	 * specific target number.
	 * 
	 * The function twoSum should return indices of the two numbers such that
	 * they add up to the target, where index1 must be less than index2. Please
	 * note that your returned answers (both index1 and index2) are not
	 * zero-based.
	 * 
	 * You may assume that each input would have exactly one solution.
	 * 
	 * Input: numbers={2, 7, 11, 15}, target=9 Output: index1=1, index2=2
	 */
    public int[] twoSum(int[] numbers, int target) {
    	// result array, contains -1s as default
        int[] result = new int[2];
        result[0] = -1;
        result[1] = -1;
        
        // map that store the index of numbers
        Map<Integer, Integer> index = new HashMap<Integer, Integer>();
        
        // add every element in array
        for (int i = 0; i < numbers.length; i++) {
            index.put(numbers[i], i + 1);
        }
        
        // find the complement of every element, try to make target
        for (int i = 0; i < numbers.length; i++) {
            int complement = target - numbers[i];
            if (index.containsKey(complement)) {
                int j = index.get(complement);
                if (i + 1 != j) {
                    result[0] = i + 1;
                    result[1] = j;
                    return result;
                }
            }
        }
        
        return result;
    }
    
    /**
	 * 3Sum
	 * 
	 * Given an array S of n integers, are there elements a, b, c in S such that
	 * a + b + c = 0? Find all unique triplets in the array which gives the sum
	 * of zero.
	 * 
	 * Note: Elements in a triplet (a,b,c) must be in non-descending order. (ie,
	 * a ¡Ü b ¡Ü c) The solution set must not contain duplicate triplets. For
	 * example, given array S = {-1 0 1 2 -1 -4},
	 * 
	 * A solution set is: (-1, 0, 1) (-1, -1, 2)
	 */
    public ArrayList<ArrayList<Integer>> threeSum(int[] num) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
        Arrays.sort(num);
        
        int len = num.length;
        for (int i = 0; i < len; i++) {
            int first = num[i];
            if (i > 0 && first == num[i - 1])
                continue;

            int start = i + 1;
            int end = len - 1;
            
            while (start < end) {
                int sum = first + num[start] + num[end];
                if (sum == 0) {
                	// add result
                    ArrayList<Integer> three = new ArrayList<Integer>();
                    three.add(first);
                    three.add(num[start]);
                    three.add(num[end]);
                    result.add(three);
                    
                    // shrink range and skip duplicate
                    start++;
                    end--;
                    while (start < end && num[start] == num[start - 1])
                        start++;
                    while (start < end && num[end] == num[end + 1])
                        end--;
                } else if (sum > 0) {
                	end--;
                    while (start < end && num[end] == num[end + 1])
                        end--;
                } else {
                	start++;
                    while (start < end && num[start] == num[start - 1])
                        start++;
                }
            }
        }
        
        return result;
    }
    
    /**
	 * 3Sum Closest
	 * 
	 * Given an array S of n integers, find three integers in S such that the
	 * sum is closest to a given number, target. Return the sum of the three
	 * integers. You may assume that each input would have exactly one solution.
	 * 
	 * For example, given array S = {-1 2 1 -4}, and target = 1.
	 * 
	 * The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
	 */
    public int threeSumClosest(int[] num, int target) {
        Arrays.sort(num);
        
        int sum = num[0] + num[1] + num[2];
        
        // traverse the array for every possible position of first number
        for (int i = 0; i < num.length - 2; i++) {
            // second number start from the one next to first one
            // third number start from the last number in the array
            for (int j = i + 1, k = num.length - 1; j < k;) {
                int temp = num[i] + num[j] + num[k];
                
                // compare temp with target
                if (temp == target) {
                    return temp;
                } else {
                    // update sum
                    if (Math.abs(temp - target) < Math.abs(sum - target)) {
                        sum = temp;
                    }
                    
                    //update j and k
                    if (temp > target) {
                        k--;
                    } else {
                        j++;
                    }
                }
            }
        }
        
        return sum;
    }
    
    /**
	 * 4Sum
	 * 
	 * Given an array S of n integers, are there elements a, b, c, and d in S
	 * such that a + b + c + d = target? Find all unique quadruplets in the
	 * array which gives the sum of target.
	 * 
	 * Note: Elements in a quadruplet (a,b,c,d) must be in non-descending order.
	 * (ie, a ¡Ü b ¡Ü c ¡Ü d) The solution set must not contain duplicate
	 * quadruplets.
	 * 
	 * For example, given array S = {1 0 -1 0 -2 2}, and target = 0.
	 * A solution set is: (-1, 0, 0, 1) (-2, -1, 1, 2) (-2, 0, 0, 2)
	 */
	public ArrayList<ArrayList<Integer>> fourSum(int[] num, int target) {
		int len = num.length;
        Map<Integer, ArrayList<ArrayList<Integer>>> map = new HashMap<Integer, ArrayList<ArrayList<Integer>>>();
        Set<ArrayList<Integer>> set = new HashSet<ArrayList<Integer>>();

        for (int i = 0; i < len; i++) {
            for (int j = i + 1; j < len; j++) {
                int sum = num[i] + num[j];
                ArrayList<Integer> two = new ArrayList<Integer>();
                two.add(i);
                two.add(j);
                
                if (map.containsKey(sum)) {
                    map.get(sum).add(two);
                } else {
                	ArrayList<ArrayList<Integer>> list = new ArrayList<ArrayList<Integer>>();
                	list.add(two);
                    map.put(sum, list);
                }
            }
        }
        
        for (int i = 0; i < len; i++) {
            for (int j = i + 1; j < len; j++) {
                int sum = num[i] + num[j];
                int target2 = target - sum;
                
                if (map.containsKey(target2)) {
                    ArrayList<ArrayList<Integer>> two = map.get(target2);
                    for (ArrayList<Integer> list : two) {
                        int x = list.get(0);
                        int y = list.get(1);
                        if (x == i || x == j || y == i || y == j) break;
                        int[] temp = new int[4];
                        temp[0] = num[i];
                        temp[1] = num[j];
                        temp[2] = num[x];
                        temp[3] = num[y];
                        Arrays.sort(temp);
                        
                        ArrayList<Integer> four = new ArrayList<Integer>();
                        for (int z : temp)
                            four.add(z);
                        
                        set.add(four);
                    }
                }
            }
        }
        
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>(set);
        return result;
    }
	
	
	
	public void test() {
		double a = Double.MAX_VALUE;
		System.out.println(a);
	}
	
	public static void main(String[] args) {
		Basic basic = new Basic();
		basic.test();
	}
}
