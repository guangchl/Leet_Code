package all;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
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
