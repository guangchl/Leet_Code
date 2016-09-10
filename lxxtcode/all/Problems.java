package all;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;

public class Problems {
	// ********************** HELPER CLASS AND FUNCTIONS **********************
	/** Definition for binary tree */
	public class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;

		TreeNode(int x) {
			val = x;
		}
	}

	public class ListNode {
		int val;
		ListNode next;

		ListNode(int x) {
			val = x;
			next = null;
		}
	}

	public ListNode arrayToList(int[] A) {
		if (A.length == 0) {
			return null;
		}

		ListNode head = new ListNode(A[0]);
		ListNode iter = head;

		for (int i = 1; i < A.length; i++) {
			iter.next = new ListNode(A[i]);
			iter = iter.next;
		}

		return head;
	}

	public void printList(ListNode head) {
		while (head != null) {
			System.out.print(head.val + " -> ");
			head = head.next;
		}
		System.out.println("null");
	}

	/** Definition for binary tree with next pointer */
	public class TreeLinkNode {
		int val;
		TreeLinkNode left, right, next;

		TreeLinkNode(int x) {
			val = x;
		}
	}
	
	/** Definition for a point. */
	class Point {
		int x;
		int y;

		Point() {
			x = 0;
			y = 0;
		}

		Point(int a, int b) {
			x = a;
			y = b;
		}
	}
	
	/** Definition for singly-linked list with a random pointer. */
	class RandomListNode {
		int label;
		RandomListNode next, random;
		RandomListNode(int x) { this.label = x; }
	};
	
	/** Definition for an interval. */
	public class Interval {
		int start;
		int end;
		Interval() { start = 0; end = 0; }
		Interval(int s, int e) { start = s; end = e; }
	}
	

	/** Definition for undirected graph. */
	class UndirectedGraphNode {
		int label;
		ArrayList<UndirectedGraphNode> neighbors;

		UndirectedGraphNode(int x) {
			label = x;
			neighbors = new ArrayList<UndirectedGraphNode>();
		}
	}
	  
	
	// ****************************** SOLUTIONS ******************************

	/**
	 * Reverse Integer
	 * 
	 * Reverse digits of an integer.
	 */
	public int reverseInteger(int x) {
		int result = 0;
		while (x != 0) {
			result *= 10;
			result += x % 10;
			x /= 10;
		}
		return result;
	}

	/**
	 * Populating Next Right Pointers in Each Node
	 * 
	 * Populate each next pointer to point to its next right node. If there is
	 * no next right node, the next pointer should be set to NULL. Initially,
	 * all next pointers are set to NULL.
	 * 
	 * Note: You may only use constant extra space. You may assume that it is a
	 * perfect binary tree (ie, all leaves are at the same level, and every
	 * parent has two children).
	 * 
	 * For example, 1 -> NULL / \ 2 -> 3 -> NULL / \ / \ 4->5->6->7 -> NULL
	 */
	public void connectPerfect(TreeLinkNode root) {
		connectChildrenPairs(root);
		connectMiddlePairs(root);
	}

	public void connectChildrenPairs(TreeLinkNode root) {
		if (root == null || root.left == null) {
			return;
		} else {
			root.left.next = root.right;
			connectChildrenPairs(root.left);
			connectChildrenPairs(root.right);
			return;
		}
	}

	public void connectMiddlePairs(TreeLinkNode root) {
		if (root == null || root.left == null) {
			return;
		}

		// connect all middle links
		TreeLinkNode left = root.left.right;
		TreeLinkNode right = root.right.left;
		while (left != null) {
			left.next = right;
			left = left.right;
			right = right.left;
		}

		// recursive in-order traverse the tree
		connectMiddlePairs(root.left);
		connectMiddlePairs(root.right);
	}

	/**
	 * Populating Next Right Pointers in Each Node II
	 * 
	 * Follow up for problem "Populating Next Right Pointers in Each Node".
	 * 
	 * What if the given tree could be any binary tree? Would your previous
	 * solution still work?
	 * 
	 * Note: You may only use constant extra space.
	 */
	public void connect(TreeLinkNode root) {
        TreeLinkNode head = root;

        // traverse the tree in level order
        while (head != null) {
            // start from the first one of former level
            TreeLinkNode parent = head;
            TreeLinkNode current = null;
            
            // traverse every child of node in this level
            while (parent != null) {
                // left child exists
                if (parent.left != null) {
                    if (current == null) { // no node in next level found yet
                        current = parent.left;
                        head = current;
                    } else {
                        current.next = parent.left;
                        current = current.next;
                    }
                }
                
                // right child exists
                if (parent.right != null) {
                    if (current == null) { // no node in next level found yet
                        current = parent.right;
                        head = current;
                    } else {
                        current.next = parent.right;
                        current = current.next;
                    }
                }
                
                // update parent
                parent = parent.next;
            }
            
            // update head
            if (current == null) {
                head = null;
            }
        }
    }

	/**
	 * Remove Element
	 * 
	 * Given an array and a value, remove all instances of that value in place
	 * and return the new length.
	 * 
	 * The order of elements can be changed. It doesn't matter what you leave
	 * beyond the new length.
	 * 
	 * Simple optimization: instead of left shift every time, move the last one
	 * to the current index.
	 */
	public int removeElement(int[] A, int elem) {
		int size = A.length;

		if (size == 0) {
			return size;
		}

		for (int i = 0; i < size; i++) {
			if (A[i] == elem) {
				A[i] = A[size - 1];
				i--;
				size--;
			}
		}

		return size;
	}

	/**
	 * Convert Sorted Array to Binary Search Tree
	 * 
	 * Given an array where elements are sorted in ascending order, convert it
	 * to a height balanced BST.
	 */
	public TreeNode sortedArrayToBST(int[] num) {
		if (num.length == 0) {
			return null;
		}
		return sortedArrayToBST(num, 0, num.length - 1);
	}

	public TreeNode sortedArrayToBST(int[] num, int start, int end) {
		if (end == start) {
			return new TreeNode(num[start]);
		} else if (end - start == 1) {
			TreeNode root = new TreeNode(num[start]);
			root.right = new TreeNode(num[end]);
			return root;
		} else if (end - start == 2) {
			TreeNode root = new TreeNode(num[start + 1]);
			root.left = new TreeNode(num[start]);
			root.right = new TreeNode(num[end]);
			return root;
		}

		int mid = (start + end) / 2;
		TreeNode root = new TreeNode(num[mid]);
		root.left = sortedArrayToBST(num, start, mid - 1);
		root.right = sortedArrayToBST(num, mid + 1, end);

		return root;
	}
	
	/**
	 * Pascal's Triangle
	 * 
	 * Given numRows, generate the first numRows of Pascal's triangle.
	 */
	public ArrayList<ArrayList<Integer>> generate(int numRows) {
		ArrayList<ArrayList<Integer>> pascal = new ArrayList<ArrayList<Integer>>();

		if (numRows == 0) {
			return pascal;
		}

		ArrayList<Integer> firstRow = new ArrayList<Integer>();
		firstRow.add(1);
		pascal.add(firstRow);

		for (int i = 2; i <= numRows; i++) {
			ArrayList<Integer> prevRow = pascal.get(i - 2);
			ArrayList<Integer> row = new ArrayList<Integer>(i);

			row.add(1);

			for (int j = 1; j < i - 1; j++) {
				row.add(prevRow.get(j - 1) + prevRow.get(j));
			}

			row.add(1);

			pascal.add(row);
		}

		return pascal;
	}
	
	/**
	 * Pascal's Triangle II
	 * 
	 * Given an index k, return the kth row of the Pascal's triangle.
	 * 
	 * For example, given k = 3, Return [1,3,3,1].
	 * 
	 * Note: Could you optimize your algorithm to use only O(k) extra space?
	 */
	public ArrayList<Integer> getRow(int rowIndex) {
        ArrayList<Integer> row = new ArrayList<Integer>();
        
        row.add(1);
        if (rowIndex == 0) {
            return row;
        }
        
        row.add(1);
        if (rowIndex == 1) {
            return row;
        }
        
        for (int i = 2; i <= rowIndex; i++) {
            ArrayList<Integer> newRow = new ArrayList<Integer>();
            
            newRow.add(1);
            
            for (int j = 1; j < i; j++) {
                newRow.add(row.get(j - 1) + row.get(j));
            }
            
            newRow.add(1);
            
            row = newRow;
        }
        
        return row;
    }

	/**
	 * Gray Code
	 * 
	 * The gray code is a binary numeral system where two successive
	 * values differ in only one bit.
	 * 
	 * Given a non-negative integer n representing the total number of bits in
	 * the code, print the sequence of gray code. A gray code sequence must
	 * begin with 0.
	 * 
	 * For example, given n = 2, return [0,1,3,2]. Its gray code sequence is:
	 * 00 - 0
	 * 01 - 1
	 * 11 - 3
	 * 10 - 2
	 */
	public ArrayList<Integer> grayCode(int n) {
		// initialize an ArrayList with length 2^n
        ArrayList<Integer> code = new ArrayList<Integer>(1 << n);
        
        // add initial code
        code.add(0);

        // add derivative code
        for (int i = 0; i < n; i++) {
            int addend = 1 << i;
            for (int j = code.size() - 1; j >= 0; j--) {
                code.add(code.get(j) + addend);
            }
        }
        
        return code;
    }
	
	/**
	 * Roman to Integer
	 * 
	 * Given a Roman numeral, convert it to an integer.
	 * 
	 * Input is guaranteed to be within the range from 1 to 3999.
	 * 
	 * I(1), V(5), X(10), L(50), C(100), D(500), M(1000)
	 * 
	 * the following solution is copied from leetcode discussion
	 */
	public int romanToInt(String s) {
		if (s.length() == 0) return 0;
		
		// map the characters in Roman number to corresponding value
        Map<Character, Integer> map = new HashMap<Character, Integer>();
        map.put('I', 1);
        map.put('V', 5);
        map.put('X', 10);
        map.put('L', 50);
        map.put('C', 100);
        map.put('D', 500);
        map.put('M', 1000);
    
        int n = s.length();
        int sum = map.get(s.charAt(n-1));
        // calculate each character in reverse order
        for (int i = n - 2; i >= 0; i--) {
			// clean and beautiful logic: characters should be in ascending
			// order from right to left, except for prefix
            if (map.get(s.charAt(i+1)) <= map.get(s.charAt(i)))
                sum += map.get(s.charAt(i));
            else
                sum -= map.get(s.charAt(i));
        }
        
        return sum;
	}

	/**
	 * Integer to Roman
	 * 
	 * Given an integer, convert it to a Roman numeral.
	 * 
	 * Input is guaranteed to be within the range from 1 to 3999.
	 * 
	 * I(1), V(5), X(10), L(50), C(100), D(500) M(1000)
	 */
	public String intToRoman(int num) {
		int thousand = num / 1000;
		int hundred = (num % 1000) / 100;
		int ten = (num % 100) / 10;
		int one = (num % 10);
		
		StringBuffer sb = new StringBuffer();

		// thousand
		for (int i = 0; i < thousand; i++) {
			sb.append("M");
		}
		
		// hundred
		if (hundred == 9) {
			sb.append("CM");
		} else if (hundred == 4) {
			sb.append("CD");
		} else {
			if (hundred > 4) {
				sb.append("D");
				hundred -=5;
			}
			for (int i = 0; i < hundred; i++) {
				sb.append("C");
			}
		}
		
		// ten
		if (ten == 9) {
			sb.append("XC");
		} else if (ten == 4) {
			sb.append("XL");
		} else {
			if (ten > 4) {
				sb.append("L");
				ten -=5;
			}
			for (int i = 0; i < ten; i++) {
				sb.append("X");
			}
		}
		
		// one
		if (one == 9) {
			sb.append("IX");
		} else if (one == 4) {
			sb.append("IV");
		} else {
			if (one > 4) {
				sb.append("V");
				one -=5;
			}
			for (int i = 0; i < one; i++) {
				sb.append("I");
			}
		}

		return sb.toString();
	}

	/**
	 * Generate Parentheses
	 * 
	 * Given n pairs of parentheses, write a function to generate all
	 * combinations of well-formed parentheses.
	 * 
	 * For example, given n = 3, a solution set is:
	 * "((()))", "(()())", "(())()", "()(())", "()()()"
	 */
	public ArrayList<String> generateParenthesis(int n) {
        ArrayList<String> result = new ArrayList<String>();
        
        StringBuffer sb = new StringBuffer();

        parenthesisRecursive(n, n, sb, result);
        
        return result;
    }
	
	public void parenthesisRecursive(int openStock, int closeStock, StringBuffer sb, ArrayList<String> result) {
		// if no "(" and ")" left, done with one combination
		if (openStock == 0 && closeStock == 0) {
			result.add(sb.toString());
			return;
		}
		
		// if still have "(" in stock
		if (openStock > 0) {
			sb.append("(");
			parenthesisRecursive(openStock - 1, closeStock, sb, result);
			sb.deleteCharAt(sb.length() - 1);
		}
		
		// if still have ")" in stock and in a valid position
		if (closeStock > openStock) {
			sb.append(")");
			parenthesisRecursive(openStock, closeStock - 1, sb, result);
			sb.deleteCharAt(sb.length() - 1);
		}
	}
	
	/**
	 * Valid Parentheses
	 * 
	 * Given a string containing just the characters '(', ')', '{', '}', '[' and
	 * ']', determine if the input string is valid.
	 * 
	 * The brackets must close in the correct order, "()" and "()[]{}" are all
	 * valid but "(]" and "([)]" are not.
	 */
	public boolean isValid(String s) {
        Stack<Character> trace = new Stack<Character>();
        
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            
            if (c == '(' || c == '[' || c == '{') {
                trace.push(c);
            } else {
                if (trace.isEmpty()) {
                    return false;
                }
                
                char cc = trace.pop().charValue();
                
                switch (cc) {
                    case '(':
                        if (c != ')') {
                            return false;
                        }
                        break;
                    case '[':
                        if (c != ']') {
                            return false;
                        }
                        break;
                    case '{':
                        if (c != '}') {
                            return false;
                        }
                        break;
                    default:
                        continue;
                }
            }
        }
        
        if (trace.isEmpty()) {
            return true;
        } else {
            return false;
        }
    }
	
	/**
	 * Longest Valid Parentheses
	 * 
	 * Given a string containing just the characters '(' and ')', find the
	 * length of the longest valid (well-formed) parentheses substring.
	 * 
	 * An example is ")()())", where the longest valid parentheses substring is
	 * "()()", which has length = 4.
	 */
	public int longestValidParentheses(String s) {
        int max = 0;

        Stack<Integer> stack = new Stack<Integer>();

        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(' || stack.isEmpty() || s.charAt(stack.peek()) == ')') {
                stack.push(i);
            } else {
                stack.pop();
                int lastEnd = stack.isEmpty() ? -1 : stack.peek();
                max = Math.max(max, i - lastEnd);
            }
        }
        
        return max;
    }
	
	/**
	 * Rotate Image
	 * 
	 * You are given an n x n 2D matrix representing an image. Rotate the image
	 * by 90 degrees (clockwise).
	 * 
	 * Follow up: Could you do this in-place?
	 */
	public void rotate(int[][] matrix) {
		int n = matrix.length;

		// rotate every circle level by level from outside to inside
		for (int i = 0; i < n / 2; i++) {
		    int boundary = n - i - 1;
		    
		    for (int j = i; j < boundary; j++) {
		        int temp = matrix[i][j];
		        matrix[i][j] = matrix[n - j - 1][i];
		        matrix[n - j - 1][i] = matrix[boundary][n - j - 1];
		        matrix[boundary][n - j - 1] = matrix[j][boundary];
		        matrix[j][boundary] = temp;
		    }
		}
	}
	
	/**
	 * Plus One
	 * 
	 * Given a number represented as an array of digits, plus one to the number.
	 */
	public int[] plusOne(int[] digits) {
        for (int i = digits.length - 1; i >= 0; i--) {
            if (digits[i] < 9) {
                digits[i]++;
                return digits;
            } else {
                digits[i] = 0;
                continue;
            }
        }
        
        int[] result = new int[digits.length + 1];
        result[0] = 1;
        
        return result;
    }
	
	/**
	 * Set Matrix Zeroes
	 * 
	 * Given a m x n matrix, if an element is 0, set its entire row and column
	 * to 0. Do it in place.
	 */
	public void setZeroes(int[][] matrix) {
		boolean firstRow = false;
        boolean firstColumn = false;
        
        // detect if there is 0 at first row
        for (int i = 0; i < matrix[0].length; i++) {
            if (matrix[0][i] == 0) {
                firstRow = true;
                break;
            }
        }
        
        // detect if there is 0 at first column
        for (int i = 0; i < matrix.length; i++) {
            if (matrix[i][0] == 0) {
                firstColumn = true;
                break;
            }
        }
        
        // if an element is 0, set first elements of that row and column to 0
        for (int i = 1; i < matrix.length; i++) {
            for (int j = 1; j < matrix[0].length; j++) {
                if (matrix[i][j] == 0) {
                    matrix[0][j] = 0;
                    matrix[i][0] = 0;
                }
            }
        }
        
        // set 0 rows
        for (int i = 1; i < matrix.length; i++) {
            if (matrix[i][0] == 0) {
                for (int j = 1; j < matrix[0].length; j++) {
                    matrix[i][j] = 0;
                }
            }
        }
        
        // set 0 columns
        for (int i = 1; i < matrix[0].length; i++) {
            if (matrix[0][i] == 0) {
                for (int j = 1; j < matrix.length; j++) {
                    matrix[j][i] = 0;
                }
            }
        }
        
        // set first row
        if (firstRow == true) {
            for (int i = 0; i < matrix[0].length; i++) {
                matrix[0][i] = 0;
            }
        }
        
        // set first column
        if (firstColumn == true) {
            for (int i = 0; i < matrix.length; i++) {
                matrix[i][0] = 0;
            }
        }
    }
	
	/**
	 * Container With Most Water
	 * 
	 * Given n non-negative integers a1, a2, ..., an, where each represents a
	 * point at coordinate (i, ai). n vertical lines are drawn such that the two
	 * endpoints of line i is at (i, ai) and (i, 0). Find two lines, which
	 * together with x-axis forms a container, such that the container contains
	 * the most water.
	 * 
	 * Note: You may not slant the container.
	 * 
	 * This solution is incredibly intelligent! From discussion on leetcode.
	 */
	public int maxArea(int[] height) {
		int maxArea = 0;
        int left = 0;
        int right = height.length - 1;
        
        while (right > left) {
            maxArea = Math.max(maxArea, (right - left) * Math.min(height[left], height[right]));
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        
        return maxArea;
	}

	/**
	 * Path Sum
	 * 
	 * Given a binary tree and a sum, determine if the tree has a root-to-leaf
	 * path such that adding up all the values along the path equals the given
	 * sum.
	 */
	public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }
        
        if (root.left == null && root.right == null) {
            return sum == root.val;
        }
        
        boolean left = false;
        if (root.left != null) {
            left = hasPathSum(root.left, sum - root.val);
        }
        
        boolean right = false;
        if (root.right != null) {
            right = hasPathSum(root.right, sum - root.val);
        }
        
        return left || right;
    }
	
	/**
	 * Path Sum II
	 * 
	 */
	public ArrayList<ArrayList<Integer>> pathSum(TreeNode root, int sum) {
        ArrayList<ArrayList<Integer>> paths = new ArrayList<ArrayList<Integer>>();
        
        if (root == null) {
            return paths;
        }
        
        pathSum(root, sum, new ArrayList<Integer>(), paths);
        
        return paths;
    }
    
    public void pathSum(TreeNode root, int sum, ArrayList<Integer> path, ArrayList<ArrayList<Integer>> paths) {
        path.add(root.val);
        sum -= root.val;
        
        if (root.left == null && root.right == null && sum == 0) {
            paths.add(path);
            return;
        }
        
        if (root.left != null && root.right == null) {
            pathSum(root.left, sum, path, paths);
            
        } else if (root.left == null && root.right != null) {
            pathSum(root.right, sum, path, paths);
            
        } else if (root.left != null && root.right != null) {
            pathSum(root.left, sum, new ArrayList<Integer>(path), paths);
            pathSum(root.right, sum, path, paths);
        }
    }

	/**
	 * Spiral Matrix
	 * 
	 * Given a matrix of m x n elements (m rows, n columns), return all elements
	 * of the matrix in spiral order.
	 */
	public ArrayList<Integer> spiralOrder(int[][] matrix) {
		ArrayList<Integer> list = new ArrayList<Integer>();
        
        int m = matrix.length;
        if (m == 0) {
        	return list;
        }
        
        int n = matrix[0].length;
        if (n == 0) {
            return list;
        }
        
        // find the boudary of the innermost loop
        int center = Math.min((m - 1) / 2, (n - 1) / 2);
        
        // adding in spiral order
        for(int i = 0; i <= center; i++) {
            
            // only one row or column elements in the loop
            if (i == m - i - 1 && i == n - i - 1) {
                list.add(matrix[i][i]);
                
            } else if (i == m - i - 1) { // only one row
                // add the row
                for (int j = i; j < n - i; j++) {
                    list.add(matrix[i][j]);
                }
                
            } else if (i == n - i - 1) { // only one column
                // add the column
                for (int j = i; j < m - i; j++) {
                    list.add(matrix[j][i]);
                }
                
            } else { // more than one element in the loop
                // upper edge
                for (int j = i; j < n - i - 1; j++) {
                    list.add(matrix[i][j]);
                }
                
                // right edge
                for (int j = i; j < m - i - 1; j++) {
                    list.add(matrix[j][n - i - 1]);
                }
                
                // bottom edge
                for (int j = n - i - 1; j > i; j--) {
                    list.add(matrix[m - i - 1][j]);
                }
                
                // left edge
                for (int j = m - i - 1; j > i; j--) {
                    list.add(matrix[j][i]);
                }
            }
        }
        
        return list;
    }
	
	/**
	 * Spiral Matrix II
	 * 
	 * Given an integer n, generate a square matrix filled with elements from 1
	 * to n2 in spiral order.
	 */
	public int[][] generateMatrix(int n) {
		int[][] matrix = new int[n][n];
        
        int num = 1;
        
        // traverse the matrix in spiral order
        for (int i = 0; i <= (n - 1) / 2; i++) {
            
            // boundary
            int b = n - i - 1;
            
            // there is only one element in the loop
            if (i == b) {
                matrix[i][i] = num++;
                
            } else { // more than one elements in the loop
                // upper edge
                for (int j = i; j < b; j++) {
                    matrix[i][j] = num++;
                }
                
                // right edge
                for (int j = i; j < b; j++) {
                    matrix[j][b] = num++;
                }
                
                // bottom edge
                for (int j = b; j > i; j--) {
                    matrix[b][j] = num++;
                }
                
                // left edge
                for (int j = b; j > i; j--) {
                    matrix[j][i] = num++;
                }
            }
        }
        
        return matrix;
    }

	/**
	 * Palindrome Number
	 * 
	 * Determine whether an integer is a palindrome. Do this in constant space.
	 */
	public boolean isPalindrome(int x) {
        // negative number can't be palindrome
		if (x < 0) {
			return false;
		}

		// single digit number must be palindrome
		if (x < 10) {
			return true;
		}

		// last digit can't be 0, since number 0 is included in former case
		if (x % 10 == 0) {
			return false;
		}

        int temp = x;
		int y = 0;
		while (temp != 0) {
			y = 10 * y + temp % 10;
			temp /= 10;
		}

		return x == y;
    }
	
	/**
	 * Sum Root to Leaf Numbers
	 * 
	 * Given a binary tree containing digits from 0-9 only, each root-to-leaf
	 * path could represent a number.
	 * 
	 * An example is the root-to-leaf path 1->2->3 which represents the number
	 * 123.
	 * 
	 * Find the total sum of all root-to-leaf numbers.
	 */
	public int sumNumbers(TreeNode root) {
        if (root == null) {
            return 0;
        }
        
        return sumNumbers(root, 0);
    }
    
    public int sumNumbers(TreeNode root, int num) {
        num *= 10;
        num += root.val;
        
        if (root.left == null && root.right == null) {
            return num;
        }
        
        int left = 0;
        if (root.left != null) {
            left = sumNumbers(root.left, num);
        }
        
        int right = 0;
        if (root.right != null) {
            right = sumNumbers(root.right, num);
        }
        
        return left + right;
    }
    
    /**
	 * Trapping Rain Water
	 * 
	 * Given n non-negative integers representing an elevation map where the
	 * width of each bar is 1, compute how much water it is able to trap after
	 * raining.
	 * 
	 * For example, Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6.
	 */
    public int trap(int[] A) {
        if (A.length < 3) {
            return 0;
        }
        
        int sum = 0;
        
        int[] h = new int[A.length];
        h[0] = 0;
        h[A.length - 1] = 0;
        
        // update the left highest border
        int highest = 0;
        for (int i = 1; i < A.length - 1; i++) {
            highest = Math.max(highest, A[i - 1]);
            h[i] = highest;
        }
        
        // update the right highest border
        highest = 0;
        for (int i = A.length - 2; i > 0; i--) {
            highest = Math.max(highest, A[i + 1]);
            // choose the lower border between left and right
            h[i] = Math.min(h[i], highest);
        }
        
        // calculate the heights of the water and add them together
        for (int i = 1; i < A.length - 1; i++) {
            h[i] = Math.max(h[i] - A[i], 0);
            sum += h[i];
        }
        
        return sum;
    }

    /**
	 * Length of Last Word
	 * 
	 * Given a string s consists of upper/lower-case alphabets and empty space
	 * characters ' ', return the length of last word in the string.
	 * 
	 * If the last word does not exist, return 0.
	 * 
	 * Note: A word is defined as a character sequence consists of non-space
	 * characters only.
	 * 
	 * For example, Given s = "Hello World", return 5.
	 */
    public int lengthOfLastWord(String s) {
        int length = 0;
        int index = s.length() - 1;
        
        // ignore the white space at the end of the string
        while (index >= 0 && s.charAt(index) == ' ') {
            index--;
        }
        
        // calculate the length of the last word
        while (index >= 0) {
            if (s.charAt(index) == ' ') {
                break;
            } else {
                length++;
                index--;
            }
        }
        
        return length;
    }

    /**
	 * Valid Sudoku
	 * 
	 * Determine if a Sudoku is valid, according to: Sudoku Puzzles - The Rules.
	 * 
	 * The Sudoku board could be partially filled, where empty cells are filled
	 * with the character '.'.
	 */
    public boolean isValidSudoku(char[][] board) {
        // check board size
        int m = board.length;
        if (m != 9) {
            return false;
        }
        int n = board[0].length;
        if (n != 9) {
            return false;
        }
        
        // 1 to 9 array, used for duplication check
        boolean[] flag = new boolean[9];
        
        // check each row
        for (int i = 0; i < m; i++) {
            // check row i
            for (int j = 0; j < n; j++) {
                // the element is valid number
                int num = board[i][j] - 48; // '0' is 48
                if (num >= 1 && num <= 9) {
                    if (flag[num - 1] == true) {
                        return false;
                    } else {
                        flag[num - 1] = true;
                    }
                }
            }
            
            // reset the flag array to all false
            resetArray(flag);
        }
        
        // check each column
        for (int j = 0; j < n; j++) {
            // check column j
            for (int i = 0; i < m; i++) {
                // the element is valid number
                int num = board[i][j] - 48; // '0' is 48
                if (num >= 1 && num <= 9) {
                    if (flag[num - 1] == true) {
                        return false;
                    } else {
                        flag[num - 1] = true;
                    }
                }
            }
            
            // reset the flag array to all false
            resetArray(flag);
        }
        
        // check each unit
        for (int i = 0; i < m; i += 3) {
            for (int j = 0; j < n; j += 3) {
                
                // traverse 9 grids in one unit
                for (int p = i; p < i + 3; p++) {
                    for (int q = j; q < j + 3; q++) {
                        // the element is valid number
                        int num = board[p][q] - 48; // '0' is 48
                        if (num >= 1 && num <= 9) {
                            if (flag[num - 1] == true) {
                                return false;
                            } else {
                                flag[num - 1] = true;
                            }
                        }
                    }
                }
                
                // reset flag array
                resetArray(flag);
            }
        }
        
        return true;
    }
    
    public void resetArray(boolean[] A) {
        for (int i = 0; i < A.length; i++) {
            A[i] = false;
        }
    }
    
    /** solution from Mingche */
	public boolean isValidSudoku2(char[][] board) {
		for (int i = 0; i < board.length; i++) {
			for (int j = 0; j < board[0].length; j++) {
				if (board[i][j] == '.')
					continue;
				if (isValid(board, i, j) == false)
					return false;
			}
		}
		return true;
	}

	public boolean isValid(char board[][], int x, int y) {
		int i, j;
		for (i = 0; i < 9; i++)
			if (i != x && board[i][y] == board[x][y])
				return false;
		for (j = 0; j < 9; j++)
			if (j != y && board[x][j] == board[x][y])
				return false;
		for (i = 3 * (x / 3); i < 3 * (x / 3 + 1); i++)
			for (j = 3 * (y / 3); j < 3 * (y / 3 + 1); j++)
				if (i != x && j != y && board[i][j] == board[x][y])
					return false;
		return true;
	}
    
    /**
     * Sudoku Solver
     */
	public boolean solveSudoku(char board[][]) {
		for (int i = 0; i < 9; ++i)
			for (int j = 0; j < 9; ++j) {
				if ('.' == board[i][j]) {
					for (int k = 1; k <= 9; ++k) {
						board[i][j] = (char) ('0' + k);
						if (isValid(board, i, j) && solveSudoku(board))
							return true;
						board[i][j] = '.';
					}
					return false;
				}
			}
		return true;
	}

    /**
	 * Count and Say
	 * 
	 * The count-and-say sequence is the sequence of integers beginning as
	 * follows:
	 * 1, 11, 21, 1211, 111221, ...
	 * 
	 * 1 is read off as "one 1" or 11.
	 * 11 is read off as "two 1s" or 21.
	 * 21 is read off as "one 2, then one 1" or 1211.
	 * 
	 * Given an integer n, generate the nth sequence.
	 * 
	 * Note: The sequence of integers will be represented as a string.
	 */
    public String countAndSay(int n) {
        StringBuffer result = new StringBuffer("1");
        
        for (int i = 2; i <= n; i++) {
            int index = 0;
            
            // construct new StringBuffer to store new string
            StringBuffer newLine= new StringBuffer();
            
            // traverse the previous string
			while (index < result.length()) {
                // select the new number to count
                char num = result.charAt(index);
                int count = 1;
                index++;
                
                // count the number
                while (index < result.length() && result.charAt(index) == num) {
                    count++;
                    index++;
                }
                
                // append the count of the number to end of current string
                newLine.append(count);
                newLine.append(num);
            }
            
            // update the result
            result = newLine;
        }
        
        return result.toString();
    }
    
    /**
	 * Longest Consecutive Sequence
	 * 
	 * Given an unsorted array of integers, find the length of the longest
	 * consecutive elements sequence.
	 * 
	 * For example, Given [100, 4, 200, 1, 3, 2], The longest consecutive
	 * elements sequence is [1, 2, 3, 4]. Return its length: 4.
	 * 
	 * Your algorithm should run in O(n) complexity.
	 */
    public int longestConsecutive(int[] num) {
        HashSet<Integer> set = new HashSet<Integer>();
        
        // add all elements in num to the set
        for (int i = 0; i < num.length; i++) {
            set.add(num[i]);
        }
        
        int max = 0;
        
        // while the set have more elements, search left and right
        while (set.size() > 0) {
        	Iterator<Integer> iter = set.iterator();
            int n = iter.next();
            int count = 1;
            iter.remove();
            
            // search left
            for (int i = n - 1; set.contains(i); i--) {
                count++;
                set.remove(i);
            }
            
            // search right
            for (int i = n + 1; set.contains(i); i++) {
                count++;
                set.remove(i);
            }
            
            max = Math.max(count, max);
        }
        
        return max;
    }

    /**
     * Flatten Binary Tree to Linked List
     * 
     * Given a binary tree, flatten it to a linked list in-place.
     * 
     * For example, given
     *          1
     *         / \
     *        2   5
     *       / \   \
     *      3   4   6
     * The flattened tree should look like:
     * 1
     *  \
     *   2
     *    \
     *     3
     *      \
     *       4
     *        \
     *         5
     *          \
     *           6
     */
    private static TreeNode lastVisited = null;
    
    public void flattenRecursive(TreeNode root) {
        if (root == null) return;
    	
        lastVisited = root;
        TreeNode right = root.right;
        
        root.right = root.left;
        root.left = null;
        flattenRecursive(root.right);
        
        lastVisited.right = right;
        flattenRecursive(right);
    }
    
    public void flatten(TreeNode root) {
        if (root == null) return;
        
        Stack<TreeNode> stack = new Stack<TreeNode>();
        
        if (root.right != null) {
            stack.push(root.right);
        }
        
        if (root.left != null) {
            stack.push(root.left);
        }
        
        while (!stack.isEmpty()) {
            TreeNode n = stack.pop();
            
            if (n.right != null) {
                stack.push(n.right);
            }
            
            if (n.left != null) {
                stack.push(n.left);
            }
            
            root.left = null;
            root.right = n;
            root = root.right;
        }
    }
    
    /**
	 * Combination Sum
	 * 
	 * Given a set of candidate numbers (C) and a target number (T), find all
	 * unique combinations in C where the candidate numbers sums to T.
	 * 
	 * The same repeated number may be chosen from C unlimited number of times.
	 * 
	 * Note: All numbers (including target) will be positive integers. Elements
	 * in a combination (a1, a2, ... , ak) must be in non-descending order. (ie,
	 * a1 ¡Ü a2 ¡Ü ... ¡Ü ak). The solution set must not contain duplicate
	 * combinations.
	 * 
	 * For example, given candidate set 2,3,6,7 and target 7, A solution set is:
	 * [7] [2, 2, 3]
	 */
    public ArrayList<ArrayList<Integer>> combinationSum(int[] candidates, int target) {
        ArrayList<ArrayList<Integer>> lists = new ArrayList<ArrayList<Integer>>();
        
        Arrays.sort(candidates);
        combinationSum(candidates, 0, target, new ArrayList<Integer>(), lists);
        
        return lists;
    }
    
    public void combinationSum(int[] candidates, int start, int target, ArrayList<Integer> list, ArrayList<ArrayList<Integer>> lists) {
        if (target == 0) {
            lists.add(list);
            return;
        }
        
        if (start == candidates.length || target < 0) {
            return;
        }
        
        for (int i = start; i < candidates.length; i++) {
            ArrayList<Integer> newList = new ArrayList<Integer>(list);
            newList.add(candidates[i]);
            combinationSum(candidates, i, target - candidates[i], newList, lists);
        }
    }
    
    /**
	 * Combination Sum II
	 * 
	 * Given a collection of candidate numbers (C) and a target number (T), find
	 * all unique combinations in C where the candidate numbers sums to T.
	 * 
	 * Each number in C may only be used once in the combination.
	 * 
	 * Note: All numbers (including target) will be positive integers. Elements
	 * in a combination (a1, a2, ... , ak) must be in non-descending order. (ie,
	 * a1 ¡Ü a2 ¡Ü ... ¡Ü ak). The solution set must not contain duplicate
	 * combinations.
	 * 
	 * For example, given candidate set 10,1,2,7,6,1,5 and target
	 * 8, A solution set is: [1, 7] [1, 2, 5] [2, 6] [1, 1, 6]
	 */
    public ArrayList<ArrayList<Integer>> combinationSum2(int[] num, int target) {
        ArrayList<ArrayList<Integer>> lists = new ArrayList<ArrayList<Integer>>();
        
        Arrays.sort(num);
        combinationSum(num, 0, target, new ArrayList<Integer>(), lists);
        
        return lists;
    }
    
    public void combinationSum2(int[] num, int start, int target, ArrayList<Integer> list, ArrayList<ArrayList<Integer>> lists) {
        if (target == 0) {
            lists.add(list);
            return;
        }
        
        if (start == num.length || target < 0) {
            return;
        }
        
        for (int i = start; i < num.length;) {
            int end = i;
            while (end + 1 < num.length && num[end + 1] == num[end]) {
                end++;
            }
            
            int savedTarget = target;
            ArrayList<Integer> addList = new ArrayList<Integer>(list);
            
            for (; i <= end; i++) {
                addList.add(num[i]);
                ArrayList<Integer> newList = new ArrayList<Integer>(addList);
                target -= num[i];
                combinationSum(num, end + 1, target, newList, lists);
            }
            
            target = savedTarget;
        }
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
	 * Add Binary
	 * 
	 * Given two binary strings, return their sum (also a binary string).
	 * 
	 * For example, a = "11" b = "1" Return "100".
	 */
    public String addBinary(String a, String b) {
        int aEnd = a.length() - 1;
        int bEnd = b.length() - 1;
        
        StringBuffer sb = new StringBuffer();
        
        int carry = 0;
        while (aEnd >= 0 || bEnd >= 0) {
            int aInt = (aEnd >= 0) ? a.charAt(aEnd--) - 48 : 0;
            int bInt = (bEnd >= 0) ? b.charAt(bEnd--) - 48 : 0;
            
            int sum = carry + aInt + bInt;
            
            carry = sum / 2;
            sb.append(sum % 2);
        }
        
        if (carry == 1) {
            sb.append(1);
        }
        
        return sb.reverse().toString();
    }
    
    /**
	 * Letter Combinations of a Phone Number
	 * 
	 * Given a digit string, return all possible letter combinations that the
	 * number could represent.
	 * 
	 * Input:Digit string "23"
	 * Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
	 */
    public ArrayList<String> letterCombinations(String digits) {
        ArrayList<String> combinations = new ArrayList<String>();
        combinations.add("");

        for (int i = 0; i < digits.length(); i++) {
            // find the character to add
            char c1, c2, c3;
            char c4 = 0;
            switch (digits.charAt(i)) {
                case '2':
                    c1 = 'a';
                    c2 = 'b';
                    c3 = 'c';
                    break;
                case '3':
                    c1 = 'd';
                    c2 = 'e';
                    c3 = 'f';
                    break;
                case '4':
                    c1 = 'g';
                    c2 = 'h';
                    c3 = 'i';
                    break;
                case '5':
                    c1 = 'j';
                    c2 = 'k';
                    c3 = 'l';
                    break;
                case '6':
                    c1 = 'm';
                    c2 = 'n';
                    c3 = 'o';
                    break;
                case '7':
                    c1 = 'p';
                    c2 = 'q';
                    c3 = 'r';
                    c4 = 's';
                    break;
                case '8':
                    c1 = 't';
                    c2 = 'u';
                    c3 = 'v';
                    break;
                case '9':
                    c1 = 'w';
                    c2 = 'x';
                    c3 = 'y';
                    c4 = 'z';
                    break;
                default:
                    c1 = 0;
                    c2 = 0;
                    c3 = 0;
            }
            
            // add new characters to old strings
            ArrayList<String> newCombinations = new ArrayList<String>();
            for (String s : combinations) {
                newCombinations.add(s + c1);
                newCombinations.add(s + c2);
                newCombinations.add(s + c3);
                if (c4 != 0) {
                    newCombinations.add(s + c4);
                }
            }
            combinations = newCombinations;
        }
        
        return combinations;
    }
    
    /**
	 * Palindrome Partitioning
	 * 
	 * Given a string s, partition s such that every substring of the partition
	 * is a palindrome.
	 * 
	 * Return all possible palindrome partitioning of s.
	 * 
	 * For example, given s = "aab", Return [ ["aa","b"], ["a","a","b"] ]
	 */
    public ArrayList<ArrayList<String>> partition(String s) {
        ArrayList<ArrayList<String>> partitions = new ArrayList<ArrayList<String>>();
        
        // separate every single letter
        ArrayList<String> singleList = new ArrayList<String>();
        for (int i = 0; i < s.length(); i++) {
            singleList.add(s.substring(i, i + 1));
        }
        partitions.add(singleList);
        
        partition(new ArrayList<String>(singleList), 0, partitions);
        return partitions;
    }
    
    public void partition(ArrayList<String> list, int start, ArrayList<ArrayList<String>> partitions) {
        for (int i = start; i < list.size() - 1; i++) {
            // search combinations at every possible length
            for (int j = i + 1; j < list.size(); j++) {
                // test validness
                int l = i, r = j;
                while (l < r) {
                    if (!areSymmetric(list.get(l), list.get(r))) {
                        break;
                    }
                    l++;
                    r--;
                }
                
                // add new case if valid
                if (l >= r) {
                    ArrayList<String> newList = new ArrayList<String>(list.subList(0, i));
                    String s = "";
                    for (int k = i; k <= j; k++) {
                        s += list.get(k);
                    }
                    newList.add(s);
                    if (j + 1 < list.size()) {
                        newList.addAll(list.subList(j + 1, list.size()));
                    }
                    partitions.add(newList);
                    partition(new ArrayList<String>(newList), i, partitions);
                }
            }
        }
    }
    
    public boolean areSymmetric(String s1, String s2) {
        if (s1.length() != s2.length()) return false;

        int len = s1.length();
        for (int i = 0; i < len; i++) {
            if (s1.charAt(i) != s2.charAt(len - i - 1)) return false;
        }
        
        return true;
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
	public ArrayList<String> wordBreak2(String s, Set<String> dict) {
		ArrayList<String> result = new ArrayList<String>();

		if (!WordBreak2(s, dict))
			return result;

		Set<Node> pending = new HashSet<Node>();

		Node root = new Node("", 0);
		pending.add(root);

		while (pending.size() > 0) {
			Set<Node> newPending = new HashSet<Node>();

			for (Node node : pending) {
				for (String word : dict) {
					if (s.substring(node.CurrentLength, s.length()).startsWith(
							word)) {
						String sentence = node.CurrentSentence + " " + word;
						int length = node.CurrentLength + word.length();
						if (length == s.length()) {
							result.add(sentence.substring(1, sentence.length()));
						} else {
							Node childNode = new Node(sentence, length);
							newPending.add(childNode);
						}
					}
				}
			}

			pending = newPending;
		}

		return result;
	}

	public boolean WordBreak2(String s, Set<String> dict) {
		// IMPORTANT: Please reset any member data you declared, as
		// the same Solution instance will be reused for each test case.

		if (s == null | dict == null)
			return false;
		boolean[] dp = new boolean[s.length() + 1];
		dp[0] = true;
		for (int i = 1; i <= s.length(); i++) {
			for (int k = 0; k < i; k++)
				if (dp[k] && dict.contains(s.substring(k, i)))
					dp[i] = true;

		}

		return dp[s.length()];
	}

	public class Node {
		public String CurrentSentence;
		public int CurrentLength;

		public Node(String sentence, int length) {
			CurrentSentence = sentence;
			CurrentLength = length;
		}
	}

	/**
	 * This is my recursive solution
	 * 
	 * This solution will also exceed time limit
	 */
	public ArrayList<String> wordBreak2Recursive(String s, Set<String> dict) {
		ArrayList<String> result = new ArrayList<String>();
		
		StringBuffer pending = new StringBuffer(s);
		
		wordBreak2Recursive(pending, 0, dict, result);
		
		return result;
	}
	
	public void wordBreak2Recursive(StringBuffer pending, int start, Set<String> dict, ArrayList<String> result) {
		for (int i = start; i < pending.length(); i++) {
			String s = pending.substring(start, i + 1);
			if (dict.contains(s)) {
				if (i == pending.length() - 1) {
					result.add(pending.toString());
				} else {
					StringBuffer newPending = new StringBuffer(pending);
					newPending.insert(i + 1, ' ');
					wordBreak2Recursive(newPending, i + 2, dict, result);
				}
			}
		}
	}
	
	/**
	 * This is my iterative solution
	 * 
	 * This solution will also exceed time limit
	 */
	public ArrayList<String> wordBreak3Iterative(String s, Set<String> dict) {
        ArrayList<String> result = new ArrayList<String>();
		
		Queue<PendingNode> queue = new LinkedList<PendingNode>();
		StringBuffer pending = new StringBuffer(s);
		queue.add(new PendingNode(pending, 0));
		
		while (!queue.isEmpty()) {
		    Queue<PendingNode> newQueue = new LinkedList<PendingNode>();
		    
		    // deal with pending on previous level
		    for (PendingNode n : queue) {
		    	StringBuffer p = n.pending;
		    	int start = n.start;
		    	
		        for (int i = start; i < p.length(); i++) {
        			String word = p.substring(start, i + 1);
        			if (dict.contains(word)) {
        				if (i == p.length() - 1) {
        					result.add(p.substring(1));
        				} else {
        					StringBuffer newPending = new StringBuffer(p);
        					newPending.insert(i + 1, ' ');
        					newQueue.add(new PendingNode(newPending, i + 2));
        				}
        			}
        		}
		    }
		    
		    // update pending queue
		    queue = newQueue;
		}

		return result;
	}
	
	public class PendingNode {
		StringBuffer pending;
		int start;
		
		public PendingNode(StringBuffer pending, int start) {
			this.pending = pending;
			this.start = start;
		}
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
        if (s.length() == 0)
            return 0;
        
        // current longest substring
        int start1 = 0, end1 = 0;
        
        // current longest end at current last one
        Set<Character> ending = new HashSet<Character>();
        int start2 = 0, end2 = 0;
        ending.add(s.charAt(0));

        for (int i = 1; i < s.length(); i++) {
            char c = s.charAt(i);
            end2 = i;
            if (ending.add(c)) {
                if (end2 - start2 >= end1 - start1) {
                    start1 = start2;
                    end1 = end2;
                }
            } else {
                for (int j = start2; j < end2; j++) {
                    if (s.charAt(j) == c) {
                        start2 = j + 1;
                        break;
                    } else {
                        ending.remove(s.charAt(j));
                    }
                }
            }
        }
        
        return end1 - start1 + 1;
    }
    
    /** This is other's solution, pretty smart, but time complexity is same */
	public int lengthOfLongestSubstring2(String s) {
		if (s == null || s.equals(""))
			return 0;
		
		int max = 0;
		int start = 0;
		int end = 0;
		boolean[] mask = new boolean[256];
		
		while (end < s.length()) {
			if (mask[(int) s.charAt(end)]) {
				mask[(int) s.charAt(start)] = false;
				start++;
			} else {
				mask[(int) s.charAt(end)] = true;
				max = Math.max(max, end - start + 1);
				end++;
			}
		}

		return max;
	}

    /**
	 * Divide Two Integers
	 * 
	 * Divide two integers without using multiplication, division and mod
	 * operator.
	 */
    public int divide(int dividend, int divisor) {
    	if (dividend == divisor)
            return 1;
        else if (divisor == Integer.MIN_VALUE)
    		return 0;
    	else if (dividend == Integer.MIN_VALUE) {
    		if (divisor < 0) 
    			return 1 + dividePositive(divisor - dividend, -divisor);
    		else
    			return -1 - dividePositive(-(dividend + divisor), divisor);
    	}

        // mark the sign
        boolean sign = false;
        if (dividend < 0 ^ divisor < 0) sign = true;
        
        dividend = Math.abs(dividend);
        divisor = Math.abs(divisor);
        
        int quotient = dividePositive(dividend, divisor);
        
        if (sign)
            return -quotient;
        else
            return quotient;
    }
    
    public int dividePositive(int dividend, int divisor) {
        int quotient = 0;

        while (dividend >= divisor) {
            int i = 1;
            int n = divisor;
            for (; n <= dividend; i <<= 1, n <<= 1) {
                dividend -= n;
                quotient += i;
                if (n > Integer.MAX_VALUE / 2)
                    break;
            }
        }
        
        return quotient;
    }

	/**
	 * Max Points on a Line
	 * 
	 * Given n points on a 2D plane, find the maximum number of points that lie
	 * on the same straight line.
	 */
    public int maxPoints(Point[] points) {
        int len = points.length;
        if (len == 0) return 0;
        
        int max = 0;
        Map<Integer, HashSet<Integer>> visitedPoints = new HashMap<Integer, HashSet<Integer>>();
        
        for (int i = 0; i < len; i++) {
            Point p = points[i];
            
            // check visited points
            if (visitedPoints.containsKey(p.x)) {
                if (visitedPoints.get(p.x).contains(p.y)) {
                    continue;
                } else {
                    visitedPoints.get(p.x).add(p.y);
                }
            } else {
                HashSet<Integer> set = new HashSet<Integer>();
                set.add(p.y);
                visitedPoints.put(p.x, set);
            }
            
            // map keep record for number of pionts on lines cross p
            Map<Double, Integer> map = new HashMap<Double, Integer>();
            int addend = 0;
            
            // pass the remaining points to construct line
            for (int j = i + 1; j < len; j++) {
                Point q = points[j];
                if (p.x == q.x && p.y == q.y) { // p == q
                    addend++;
                } else {
                    Double k;
                    if (p.x == q.x)
                        k = Double.MAX_VALUE;
                    else if (p.y == q.y)
                        k = 0.0;
                    else
                        k = (double)(q.y - p.y)/(q.x - p.x);
                    
                    if (map.containsKey(k)) {
                        map.put(k, map.get(k) + 1);
                    } else {
                        map.put(k, 2);
                    }
                }
            }
            
            // find max for this round cross p
            if (map.size() == 0) {
                max = Math.max(max, 1 + addend);
            } else {
                for (Integer newMax : map.values()) {
                    max = Math.max(max, newMax + addend);
                }
            }
        }
        
        return max;
    }
 
    /**
	 * Word Search
	 * 
	 * Given a 2D board and a word, find if the word exists in the grid.
	 * 
	 * The word can be constructed from letters of sequentially adjacent cell,
	 * where "adjacent" cells are those horizontally or vertically neighboring.
	 * The same letter cell may not be used more than once.
	 */
    public boolean exist(char[][] board, String word) {
        if(word.length() == 0)   return true;
        int h = board.length;
        if(h == 0)    return false;
        int w = board[0].length;
        boolean[][] flag = new boolean[h][w];
        
        for(int i = 0; i < h; i++){
            for(int j = 0; j < w; j++){
                if(word.charAt(0) == board[i][j]){
                    if(search(board, word, 0, w, h, j, i, flag)) return true;
                }
            }
        }
        
        return false;
    }
    
    public boolean search(char[][] board, String word, int spos, int w, int h, int x, int y, boolean[][] flag){
        if(spos == word.length())  return true;
        if(x < 0 || x >= w || y < 0 || y >= h)   return false;
        if(flag[y][x] || board[y][x] != word.charAt(spos))   return false;
        
        flag[y][x] = true;
        
        // up
        if(search(board, word, spos + 1, w, h, x, y-1, flag)){
            return true;
        }
        // down
        if(search(board, word, spos + 1, w, h, x, y+1, flag)){
            return true;
        }
        // left
        if(search(board, word, spos + 1, w, h, x-1, y, flag)){
            return true;
        }
        // right
        if(search(board, word, spos + 1, w, h, x+1, y, flag)){
            return true;
        }
        
        flag[y][x] = false;
        
        return false;
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
	 * Valid Palindrome
	 * 
	 * Given a string, determine if it is a palindrome, considering only
	 * alphanumeric characters and ignoring cases.
	 */
	public boolean isPalindrome(String s) {
        s = s.toLowerCase();
        
        int l = 0;
        int r = s.length() - 1;
        while (l < r) {
            char cl = s.charAt(l);
            if ((cl < 'a' || cl > 'z') && (cl < '0' || cl > '9')) {
                l++;
                continue;
            }
            char cr = s.charAt(r);
            if ((cr < 'a' || cr >'z') && (cr < '0' || cr > '9')) {
                r--;
                continue;
            }
            
            if (cl != cr) {
                return false;
            }
            l++;
            r--;
        }
        
        return true;
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
	 * Here is an example: S = "rabbbit", T = "rabbit"
	 * 
	 * Return 3.
	 */
	public int numDistinct(String S, String T) {
        // Note: The Solution object is instantiated only once and is reused by each test case.
        if(S == null || T == null) return -1;
        
        int[] dp = new int[T.length() + 1];
        dp[T.length()] = 1;
        
        for (int i = S.length() - 1; i >= 0; --i) {
            for (int j = 0; j < T.length(); ++j) {
                if(S.charAt(i) == T.charAt(j))
                	dp[j] += dp[j + 1];
            }
        }
        
        return dp[0];
    }

	public int numDistinctRecursive(String S, String T) {
        return numDistinctRecursive(S, 0, T, 0);
    }
    
    public int numDistinctRecursive(String S, int s, String T, int t) {
        int sLen = S.length();
        int tLen = T.length();
        if (t == tLen) {
            return 1;
        }

        int result = 0;
        
        while (s <= sLen - tLen + t) {
            if (S.charAt(s) == T.charAt(t)) {
                result += numDistinctRecursive(S, s + 1, T, t + 1);
            }
            s++;
        }

        return result;
    }
    
    /**
     * Scramble String
     */
    public boolean isScramble(String s1, String s2) {
        if(s1.length() != s2.length()){
			return false;
		}
        if(s1.length()==1 && s2.length()==1){
        	return s1.charAt(0) == s2.charAt(0);
        }
        
        char[] s1ch = s1.toCharArray();
		char[] s2ch = s2.toCharArray();
		Arrays.sort(s1ch);
		Arrays.sort(s2ch);
		if(!new String(s1ch).equals(new String(s2ch))){
			return false;
		}
		
		for(int i=1; i<s1.length(); i++){		
			String s11 = s1.substring(0, i);
			String s12 = s1.substring(i);
			String s21 = s2.substring(0, i);
			String s22 = s2.substring(i);

			if(isScramble(s11, s21) && isScramble(s12, s22)){
				return true;
			}
		
			s21 = s2.substring(0, s2.length()-i);
			s22 = s2.substring(s2.length()-i);
			if(isScramble(s11, s22) && isScramble(s12, s21)){
				return true;
			}
		}
		return false;
    }
    
    /**
     * Simplify Path
     */
    public String simplifyPath(String path) {
        String[] list = path.split("/");
        
        Stack<String> stack = new Stack<String>();
        for (String s : list) {
            if (s.equals("") || s.equals(".")) {
                continue;
            } else if (s.equals("..")) {
                if (stack.size() != 0) {
                    stack.pop();
                }
            } else {
                stack.push(s);
            }
        }
        
        StringBuffer sb = new StringBuffer();
        for (String s : stack) {
            sb.append("/").append(s);
        }
        
        return (sb.length() == 0) ? "/" : sb.toString();
    }
    
    /**
     * ZigZag Conversion
     */
    public String convert(String s, int nRows) {
        ArrayList<StringBuffer> rowList = new ArrayList<StringBuffer>();
        for (int i = 0; i < nRows; i++) {
            rowList.add(new StringBuffer());
        }
        
        int len = s.length();
        
        int i = 0;
        while (i < len) {
            // first column
            for (int j = 0; j < nRows && i < len; j++) {
                rowList.get(j).append(s.charAt(i++));
            }
            
            // diagnal
            for (int j = nRows - 2; j > 0 && i < len; j--) {
                rowList.get(j).append(s.charAt(i++));
            }
        }
        
        StringBuffer result = new StringBuffer();
        for (StringBuffer sb : rowList) {
            result.append(sb);
        }
        
        return result.toString();
    }

    /**
     * Evaluate Reverse Polish Notation
     */
    public int evalRPN(String[] tokens) {
        if (tokens == null || tokens.length == 0) {
            return 0;
        }
        
        Stack<Integer> stack = new Stack<Integer>();
        for (String s : tokens) {
            int a, b;
            
            switch (s) {
                case "+":
                    b = stack.pop();
                    a = stack.pop();
                    stack.push(a + b);
                    break;
                case "-":
                    b = stack.pop();
                    a = stack.pop();
                    stack.push(a - b);
                    break;
                case "*":
                    b = stack.pop();
                    a = stack.pop();
                    stack.push(a * b);
                    break;
                case "/":
                    b = stack.pop();
                    a = stack.pop();
                    stack.push(a / b);
                    break;
                default:
                    stack.push(Integer.valueOf(s));
            }
        }
        
        return stack.pop();
    }
    
    /**
     * Text Justification
     */
    public ArrayList<String> fullJustify(String[] words, int L) {
		ArrayList<String> result = new ArrayList<String>();
		if (words == null || words.length == 0) {
			return result;
		}

		int left = L - words[0].length();
		int start = 0;
		int end = 0;
		int spaceSlots = 0;

		for (int i = 1; i <= words.length; i++) {
			// System.out.println("for...");
			// can not add new word
			if (i == words.length || left - words[i].length() - (end - start + 1) < 0) {
				// System.out.println("  if...");
				// System.out.println("    start: " + start + " end: " + end + " i: " + i);
				// construct new line
				StringBuffer line = new StringBuffer(L);
				if (spaceSlots == 0) {
					line.append(words[start]);
					for (int j = words[start].length(); j < L; j++) {
						line.append(" ");
					}
				} else {
					for (int j = start; j < end; j++) {
						line.append(words[j]);

						if (i == words.length) {
							line.append(" ");
						} else {
							int space = left / spaceSlots;
							// System.out.println("    space: " + space);
							if (left % spaceSlots > 0) {
								space++;
							}
							
							left -= space;
							
							while (space-- > 0) {
								line.append(" ");
							}
						}

						spaceSlots--;
					}
					line.append(words[end]);
					for (int j = line.length(); j < L; j++) {
						line.append(" ");
					}
				}
				// System.out.println("    " + line.toString() + "\n");
				result.add(line.toString());

				// start next round
				if (i < words.length) {
					start = i;
					end = i;
					left = L - words[i].length();
					spaceSlots = 0;
				}
			} else {
				// System.out.println("  else...");
			 	// System.out.println("    " + words[i]);
				left -= words[i].length();
				end = i;
				spaceSlots++;
			}
		}

		return result;
    }
    
    /** Other's solution */
	public ArrayList<String> fullJustify2(String[] words, int L) {
		ArrayList<String> list = new ArrayList<String>();
		ArrayList<String> res = new ArrayList<String>();
		StringBuilder sb = new StringBuilder();
		int cur = 0;
		int len = 0;

		while (cur < words.length) {
			sb.setLength(0);
			sb.append(words[cur]);

			list.clear();
			len = words[cur].length();
			cur++;

			while (cur < words.length && len + 1 + words[cur].length() <= L) {
				list.add(" " + words[cur]);
				len += words[cur].length() + 1;

				cur++;

			}
			if (cur < words.length && list.size() > 0) {
				int spaces = L - len;
				int avg = spaces / list.size();
				int rem = spaces % list.size();
				for (int i = 0; i < list.size(); i++) {
					appendSpaces(sb, i < rem ? avg + 1 : avg);
					sb.append(list.get(i));
				}
			} else {
				for (int i = 0; i < list.size(); i++)
					sb.append(list.get(i));
				// sb.append(list.get(0));
				appendSpaces(sb, L - len);
			}

			res.add(sb.toString());
		}
		return res;
	}

	private void appendSpaces(StringBuilder sb, int n) {
		for (int i = 0; i < n; i++)
			sb.append(' ');
	}
    
	/**
	 * Candy
	 */
	public int candy(int[] ratings) {
        if (ratings == null || ratings.length == 0) {
            return 0;
        }
        
        int[] candys = new int[ratings.length];
        for (int i = 0; i < candys.length; i++) {
            candys[i] = 1;
        }
        
        for (int i = 1; i < candys.length; i++) {
            if (ratings[i] > ratings[i - 1]) {
                candys[i] = candys[i - 1] + 1;
            }
        }
        
        for (int i = candys.length - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1]) {
                candys[i] = Math.max(candys[i], candys[i + 1] + 1);
            }
        }
        
        int sum = 0;
        for (int i = 0; i < candys.length; i++) {
            sum += candys[i];
        }
        
        return sum;
    }
	
	/**
	 * Merge Intervals
	 */
	public ArrayList<Interval> merge(ArrayList<Interval> intervals) {
        ArrayList<Interval> result = new ArrayList<Interval>();
        if (intervals == null || intervals.size() == 0) {
            return result;
        }
        
        Collections.sort(intervals, new Comparator<Interval>() {
                    @Override
                    public int compare(Interval interval1, Interval interval2) {
                        if (interval1.start < interval2.start) {
                            return -1;
                        } else if (interval1.start > interval2.start) {
                            return 1;
                        } else if (interval1.end < interval2.end) {
                            return -1;
                        } else if (interval1.end > interval2.end) {
                            return 1;
                        }
                        
                        return 0;
                    }
                });
        
        result.add(intervals.get(0));
        for (int i = 1; i < intervals.size(); i++) {
            Interval last = result.get(result.size() - 1);
            Interval next = intervals.get(i);
            if (last.end < next.start) {
                result.add(next);
            } else if (last.end >= next.end) {
                continue;
            } else {
                result.remove(last);
                result.add(new Interval(last.start, next.end));
            }
        }
        
        return result;
    }
	
	/**
	 * Insert Interval
	 */
	public ArrayList<Interval> insert(ArrayList<Interval> intervals, Interval newInterval) {
        if (intervals == null || newInterval == null || intervals.size() == 0) {
            if (intervals == null) {
                intervals = new ArrayList<Interval>();
            }
            
            if (intervals != null) {
                intervals.add(newInterval);
            }
            
            return intervals;
        }
        
        // newInterval beyond the origial range
        if (newInterval.start > intervals.get(intervals.size() - 1).end) {
            intervals.add(newInterval);
            return intervals;
        } else if (newInterval.end < intervals.get(0).start) {
            intervals.add(0, newInterval);
            return intervals;
        }
        
        // start and end intervals to merge
        int start = 0;
        int end = intervals.size() - 1;
        
        // search for start
        int left = 0;
        int right = intervals.size() - 1;
        while (left < right) {
            int mid = (left + right) >>> 1;
            if (intervals.get(mid).start > newInterval.start) {
                right = mid;
            } else if (intervals.get(mid).end < newInterval.start) {
                left = mid + 1;
            } else {
                start = mid;
                break;
            }
        }
        if (left == right) {
            start = left;
        }
        
        // search for end
        left = start - 1;
        right = intervals.size() - 1;
        while (left < right) {
            int mid = (left + right + 1) >>> 1;
            if (intervals.get(mid).start > newInterval.end) {
                right = mid - 1;
            } else if (intervals.get(mid).end < newInterval.end) {
                left = mid;
            } else {
                end = mid;
                break;
            }
        }
        if (left == right) {
            end = left;
        }
        System.out.println(start + " " + end);
        // insert without merge
        if (start > end) {
            intervals.add(start, newInterval);
            return intervals;
        }
        
        // insert and merge the intervals
        intervals.get(start).start = Math.min(intervals.get(start).start, newInterval.start);
        intervals.get(start).end = Math.max(intervals.get(end).end, newInterval.end);
        for (int i = end; i > start; i--) {
            intervals.remove(i);
        }
        
        return intervals;
    }

	/**
	 * Multiply Strings
	 */
	public String multiply(String num1, String num2) {
        if (num1 == null || num2 == null || num1.length() == 0 || num2.length() == 0) {
            return null;
        }
        
        if (num1.equals("0") || num2.equals("0")) {
            return "0";
        }
        
		String n1 = new StringBuilder(num1).reverse().toString();
		String n2 = new StringBuilder(num2).reverse().toString();
		int[] d = new int[n1.length() + n2.length()];
		for (int i = 0; i < n1.length(); i++) {
			for (int j = 0; j < n2.length(); j++) {
				d[i + j] += (n1.charAt(i) - '0') * (n2.charAt(j) - '0');
			}
		}
		
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < d.length - 1; i++) {
			int digit = d[i] % 10;
			int carry = d[i] / 10;
			d[i + 1] += carry;
			sb.insert(0, digit);
		}
		
		if (d[d.length - 1] != 0) {
			sb.insert(0, d[d.length - 1]);
		}

		return sb.toString();
	}
	
	/**
	 * Interleaving String
	 * 
	 * Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and
	 * s2.
	 * 
	 * For example, Given: 
	 * s1 = "aabcc", 
	 * s2 = "dbbca",
	 * 
	 * When s3 = "aadbbcbcac", return true. 
	 * When s3 = "aadbbbaccc", return false.
	 */
	public boolean isInterleave(String s1, String s2, String s3) {
        if (s1 == null || s2 == null || s3 == null) {
            return false;
        }
        
        int m = s1.length();
        int n = s2.length();
        int len = s3.length();
        if (len != m + n) {
            return false;
        }

        boolean[][] match = new boolean[m + 1][n + 1];
        match[0][0] = true;
        for (int i = 1; i <= m; i++) {
            if (s1.charAt(i - 1) == s3.charAt(i - 1)) {
                match[i][0] = true;
            } else {
                break;
            }
        }
        for (int i = 1; i <= n; i++) {
            if (s2.charAt(i - 1) == s3.charAt(i - 1)) {
                match[0][i] = true;
            } else {
                break;
            }
        }
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if ((match[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1)) || 
                    (match[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1))) {
                    match[i][j] = true;
                }
            }
        }
        
        return match[m][n];
    }
	
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
	
	public void test() {
		//int[] num = {0,0,0,0};
		//int target = 0;
		//System.out.println(fourSum(num, target));
		// int[][] matrix = {{0,0,0,5},{4,3,1,4},{0,1,1,4},{1,2,1,3},{0,0,1,1}};
		//char[][] board = {{'.','8','7','6','5','4','3','2','1'},{'2','.','.','.','.','.','.','.','.'},{'3','.','.','.','.','.','.','.','.'},{'4','.','.','.','.','.','.','.','.'},{'5','.','.','.','.','.','.','.','.'},{'6','.','.','.','.','.','.','.','.'},{'7','.','.','.','.','.','.','.','.'},{'8','.','.','.','.','.','.','.','.'},{'9','.','.','.','.','.','.','.','.'}};
		//System.out.println("mississippi\nissip");
		//System.out.println(kmp("mississippi", "issip"));
		Interval i1 = new Interval(3, 5);
		Interval i2 = new Interval(12, 15);
		Interval newInterval = new Interval(6, 6);
		ArrayList<Interval> intervals = new ArrayList<Interval>();
		intervals.add(i1);
		intervals.add(i2);
		for (Interval i : insert(intervals, newInterval)) {
			System.out.println(i);
		}
	}
	
	public static void main(String[] args) {
		Problems m = new Problems();
		m.test();
	}

}
