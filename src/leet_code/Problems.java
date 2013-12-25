package leet_code;

import java.util.ArrayList;

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

	// ****************************** SOLUTIONS ******************************
	/**
	 * Single Number
	 * 
	 * Given an array of integers, every element appears twice except for one.
	 * Find that single one. Time: O(n). Space: O(0).
	 * 
	 * If there's no space constraint, Map should be a common solution
	 */
	public int singleNumber(int[] A) {
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
//		int ones = 0, twos = 0, threes = 0;
//		for (int i = 0; i < n; i++) {
//			twos |= ones & A[i];
//			ones ^= A[i];
//			threes = ones & twos;
//			ones &= ~threes;
//			twos &= ~threes;
//		}
//		return ones;
	}

	/**
	 * Maximum Depth of Binary Tree
	 * 
	 * Given a binary tree, find its maximum depth. The maximum depth is the
	 * number of nodes along the longest path from the root node down to the
	 * farthest leaf node.
	 */
	public int maxDepth(TreeNode root) {
		if (root == null) { // base case: empty tree from root
			return 0;
		} else { // recursion
			return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
		}
	}

	/**
	 * Same Tree
	 * 
	 * Given two binary trees, write a function to check if they are equal or
	 * not. Two binary trees are considered equal if they are structurally
	 * identical and the nodes have the same value.
	 */
	public boolean isSameTree(TreeNode p, TreeNode q) {
		if (p == null && q == null) { // p and q are both empty tree
			return true;
		} else if (p == null || q == null) { // one of p and q is empty tree
			return false;
		} else { // both p and q are not empty
			return p.val == q.val && isSameTree(p.left, q.left)
					&& isSameTree(p.right, q.right);
		}
	}

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
	 * Best Time to Buy and Sell Stock
	 * 
	 * Say you have an array for which the ith element is the price of a given
	 * stock on day i. If you were only permitted to complete at most one
	 * transaction (ie, buy one and sell one share of the stock), design an
	 * algorithm to find the maximum profit.
	 */
	public int maxProfit(int[] prices) {
		
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

	}

	/**
	 * Unique Binary Search Trees
	 * 
	 * Given n, how many structurally unique BST's (binary search trees) that
	 * store values 1...n? For example: Given n = 3, there are a total of 5
	 * unique BST's.
	 */
	public int numTrees(int n) {
		// base case
		if (n == 0 || n == 1) {
			return 1;
		}
		
		// recursive traverse every possible case
		int sum = 0;
		for (int i = 0; i < n; i++) {
			sum = sum + numTrees(i) * numTrees(n - i - 1);
		}
		
		return sum;
	}
	
	/**
	 * Unique Binary Search Trees II
	 * 
	 * Given n, generate all structurally unique BST's (binary search trees)
	 * that store values 1...n.
	 */
	public ArrayList<TreeNode> generateTrees(int n) {
        return generateTrees(1, n);
    }
	
	public ArrayList<TreeNode> generateTrees(int left, int right) {
		ArrayList<TreeNode> trees = new ArrayList<TreeNode>();
		
		// base case
		if (left > right) {
			trees.add(null);
			return trees;
		}
		
		TreeNode root;
		for (int i = left; i <= right; i++) {
			for (TreeNode leftTree : generateTrees(left, i - 1)) {
				for (TreeNode rightTree : generateTrees(i + 1, right)) {
					root = new TreeNode(i);
					root.left = leftTree;
					root.right = rightTree;
					trees.add(root);
				}
			}
        }
		
		return trees;
	}
	
	/**
	 * Linked List Cycle
	 * 
	 * Given a linked list, determine if it has a cycle in it.
	 * 
	 * Follow up: Can you solve it without using extra space?
	 */
	public boolean hasCycle(ListNode head) {
		// set two runners
		ListNode slow = head;
		ListNode fast = head;
		
		// fast runner move 2 steps at one time while slow runner move 1 step,
		// if traverse to a null, there must be no loop
		while (fast != null && fast.next != null) {
			slow = slow.next;
			fast = fast.next.next;
			if (slow == fast) {
				return true;
			}
		}
		return false;
	}
	
	/**
	 * Linked List Cycle II
	 * 
	 * Given a linked list, return the node where the cycle begins. If there is
	 * no cycle, return null.
	 * 
	 * Follow up: Can you solve it without using extra space?
	 */
    public ListNode detectCycle(ListNode head) {
    	// set two runners
        ListNode slow = head;
        ListNode fast = head;
        
		// first time meet: fast runner move 2 steps at one time while slow
		// runner move 1 step,
        while (fast != null && fast.next != null) {
			slow = slow.next;
			fast = fast.next.next;
			if (slow == fast) {
				break;
			}
		}
        
        // if stopped by null, indicating no loop
        if (fast == null || fast.next == null) {
			return null;
		}
        
        // one runner start from the head, both runner move 1 step each time
        fast = head;
        while (fast != slow) {
			fast = fast.next;
			slow = slow.next;
		}
        
        return fast;
    }
    
	/**
	 * Search Insert Position
	 * 
	 * Given a sorted array and a target value, return the index if the target
	 * is found. If not, return the index where it would be if it were inserted
	 * in order. You may assume no duplicates in the array.
	 */
    public int searchInsert(int[] A, int target) {
    	// give a range to for recursion
        return binarySearchInsert(A, target, 0, A.length - 1);
    }
    
    public int binarySearchInsert(int[] A, int target, int left, int right) {
    	// base case: length of insert range == 1
    	if (left == right) {
			if (A[left] >= target) {
				return left;
			} else {
				return left + 1;
			}
		}
    	
    	// binary recursion
    	int mid = (left + right) / 2;
    	if (A[mid] == target) {
			return mid;
		} else if (A[mid] < target) {
			return binarySearchInsert(A, target, mid + 1, right);
		} else {
			if (mid == left) { // length of insert range == 2
				return mid;
			} else {
				return binarySearchInsert(A, target, left, mid - 1);
			}
		}
    }
    
	/**
	 * Remove Duplicates from Sorted List
	 * 
	 * Given a sorted linked list, delete all duplicates such that each element
	 * appear only once.
	 * 
	 * For example: Given 1->1->2, return 1->2. Given 1->1->2->3->3, return
	 * 1->2->3.
	 */
	public ListNode deleteDuplicates(ListNode head) {
		if (head != null) {
			ListNode iter = head;
			while (iter.next != null) {
				if (iter.next.val == iter.val) {
					iter.next = iter.next.next;
				} else {
					iter = iter.next;
				}
			}
		}
		return head;
	}

	/**
	 * Remove Duplicates from Sorted List II
	 * 
	 * Given a sorted linked list, delete all nodes that have duplicate numbers,
	 * leaving only distinct numbers from the original list.
	 * 
	 * For example, Given 1->2->3->3->4->4->5, return 1->2->5. Given
	 * 1->1->1->2->3, return 2->3.
	 */
	public ListNode deleteDuplicates2(ListNode head) {
		// set the dummy head
		ListNode root = new ListNode(0);
		root.next = head;
		
		ListNode prev = root;
		ListNode iter = head;
		
		while (iter != null) {
			// find the last one for each number along the list
			while (iter.next != null && iter.val == iter.next.val) {
				iter = iter.next;
			}
			
			// check duplication
			if (prev.next == iter) {
				prev = iter;
			} else {
				prev.next = iter.next;
			}
			
			// begin next round iteration
			iter = iter.next;
		}
		
		return root.next;
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
	 * For example,
	 *         1 -> NULL
	 *       /  \
	 *      2 -> 3 -> NULL
	 *     / \  / \
	 *    4->5->6->7 -> NULL
	 */
	public void connect(TreeLinkNode root) {
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
	
	public void test() {
		int[] A = { 1, 2, 3, 3, 4, 4, 5 };
		printList(arrayToList(A));
		printList(deleteDuplicates2(arrayToList(A)));
	}

	public static void main(String[] args) {
		Problems m = new Problems();
		m.test();
	}

}
