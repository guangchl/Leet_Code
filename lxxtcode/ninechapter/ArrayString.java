package ninechapter;

import java.util.HashSet;
import java.util.Set;
import java.util.Stack;

public class ArrayString {

	/**
	 * Remove Duplicates from Sorted Array
	 * @param A
	 * @return new length
	 */
	public int removeDuplicates(int[] A) {
		if (A == null || A.length == 0) {
            return 0;
        }
        
        int last = 0;
        
        for (int i = 1; i < A.length; i++) {
            if (A[i] != A[last]) {
                A[++last] = A[i];
            }
        }
        
        return last + 1;
    }
	
	/**
	 * Find Duplicate from Unsorted Array
	 * 
	 * A = [2, 3, 4, 2, 5], length = n, containing one missing, one duplicate
	 * 
	 * @param A
	 * @return new length
	 */
	public int findDup(int[] num) {
	    if (num == null || num.length == 0) {
	        return -1;
	    }

		for (int i = 0; i < num.length; i++) {
			if (num[num[i] - 1] == num[i]) {
				return num[i];
			} else {
				if (num[i] == i + 1)
					continue;
				else {
					int j = num[i] - 1;
					num[i] = num[j];
					num[j] = j + 1;
					i--;
				}
			}
		}
	    
	    return -1;
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
        if (A == null || A.length == 0) {
            return 0;
        }
        
        int max = A[0];
        int maxEndHere = A[0];
        
        for (int i = 1; i < A.length; i++) {
            maxEndHere = (maxEndHere > 0) ? (maxEndHere + A[i]) : A[i];
            max = (max > maxEndHere) ? max: maxEndHere;
        }
        
        return max;
	}
	
	public int subArrayWithValue(int[] A) {
		return 0;
	}
	
	public int subArrayCloseToValue(int[] A) {
		return 0;
	}
	
	public int maxSubMatrix(int[][] A) {
		if (A == null || A.length == 0 || A[0].length == 0) {
			return 0;
		}
		
		int max = A[0][0];
		
		for (int i = 0; i < A.length; i++) {
			for (int j = i; j < A.length; j++) {
				
			}
		}
		return 0;
	}

	/**
	 * Longest Substring Without Repeating Characters
	 *
	 * Given a string, find the length of the longest substring without
	 * repeating characters. For example, the longest substring without
	 * repeating letters for "abcabcbb" is "abc", which the length is 3. For
	 * "bbbbb" the longest substring is "b", with the length of 1.
	 */	
	public int lengthOfLongestSubstring(String s) {
		if (s == null || s.length() == 0) {
			return 0;
		}
		
		int start = 0;
		int max = 1;
		Set<Character> set = new HashSet<Character>();
		
		for (int i = 0; i < s.length(); i++) {
			while (set.contains(s.charAt(i))) {
				set.remove(s.charAt(start++));
			}
			set.add(s.charAt(i));
			max = Math.max(max, set.size());
		}
		
		return max;
	}
	
	/**
	 * Largest Rectangle in Histogram
	 * 
	 * Given n non-negative integers representing the histogram's bar height
	 * where the width of each bar is 1, find the area of largest rectangle in
	 * the histogram.
	 */
	public int largestRectangleArea(int[] height) {
		int largest = 0;

		Stack<Integer> stack = new Stack<Integer>();
		int index = 0;
		while (index < height.length || !stack.isEmpty()) {
			if (index < height.length
					&& (stack.isEmpty() || height[index] >= height[stack.peek()])) {
				stack.push(index++);
			} else {
				int end = index - 1;
				int h = height[stack.pop()];
				while (!stack.isEmpty() && height[stack.peek()] == h) {
					stack.pop();
				}
				int start = stack.isEmpty() ? -1 : stack.peek();
				largest = Math.max(largest, h * (end - start));
			}
		}

		return largest;
	}
    
	/**
	 * Given a 2D binary matrix filled with 0's and 1's, find the largest
	 * rectangle containing all ones and return its area.
	 */
	public int maximalRectangle(char[][] matrix) {
        int m = matrix.length;
        if (m == 0) return 0;
        int n = matrix[0].length;
        if (n == 0) return 0;
        int[][] rectangles = new int[m][n];
        
        // pre-process first row
        for (int i = 0; i < n; i++) {
            if (matrix[0][i] == '1') {
                rectangles[0][i] = 1;
            }
        }
        
        // pre-process other rows
        for (int i = 1; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '1')
                    rectangles[i][j] = rectangles[i - 1][j] + 1;
            }
        }
        
        int area = 0;
        
        // calculate largest rectangle for each row
        for (int i = 0; i < m; i++) {
            area = Math.max(area, largestRectangleArea(rectangles[i]));
        }
        
        return area;
    }
	
	public void test() {
		int[] A = {2, 3, 4, 2, 5};
		System.out.println(findDup(A));
	}
	
	public static void main(String[] args) {
		ArrayString as = new ArrayString();
		as.test();
	}

}
