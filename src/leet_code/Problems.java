package leet_code;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;
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
		// Notice: here we need a null to represent empty tree
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
	 * For example, 1 -> NULL / \ 2 -> 3 -> NULL / \ / \ 4->5->6->7 -> NULL
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

	/**
	 * Binary Tree Inorder Traversal
	 * 
	 * Given a binary tree, return the inorder traversal of its nodes' values.
	 * 
	 * For example: Given binary tree {1,#,2,3}, return [1, 3, 2]
	 * 
	 * Note: Recursive solution is trivial, could you do it iteratively?
	 */
	public ArrayList<Integer> inorderTraversalIterative(TreeNode root) {
		ArrayList<Integer> inList = new ArrayList<Integer>();

		// the while loop below don't handle null, so null won't be pushed in s
		if (root == null) {
			return inList;
		}

		// initial the stack which will store the TreeNodes inorderly
		Stack<TreeNode> s = new Stack<TreeNode>();
		s.push(root);

		TreeNode n;
		while (!s.isEmpty()) {
			n = s.pop();

			if (n.right != null) {
				s.push(n.right);
			}

			if (n.left != null) {
				s.push(new TreeNode(n.val));
				s.push(n.left);
			} else {
				inList.add(n.val);
			}
		}

		return inList;
	}

	public ArrayList<Integer> inorderTraversalRecursive(TreeNode root) {
		ArrayList<Integer> preList = new ArrayList<Integer>();

		if (root == null) {
			return preList;
		}

		for (Integer i : inorderTraversalRecursive(root.left)) {
			preList.add(i);
		}

		preList.add(root.val);

		for (Integer i : inorderTraversalRecursive(root.right)) {
			preList.add(i);
		}

		return preList;
	}

	/**
	 * Binary Tree Preorder Traversal
	 * 
	 * Given a binary tree, return the preorder traversal of its nodes' values.
	 * 
	 * For example: Given binary tree {1,#,2,3}, return [1, 2, 3]
	 * 
	 * Note: Recursive solution is trivial, could you do it iteratively?
	 */
	public ArrayList<Integer> preorderTraversalIterative(TreeNode root) {
		ArrayList<Integer> preList = new ArrayList<Integer>();

		// the while loop below don't handle null, so null won't be pushed in s
		if (root == null) {
			return preList;
		}

		// initial the stack which will store the TreeNodes preorderly
		Stack<TreeNode> s = new Stack<TreeNode>();
		s.push(root);

		TreeNode n;
		while (!s.isEmpty()) {
			n = s.pop();

			if (n.right != null) {
				s.push(n.right);
			}

			if (n.left != null) {
				s.push(n.left);
			}

			preList.add(n.val);
		}

		return preList;
	}

	public ArrayList<Integer> preorderTraversalRecursive(TreeNode root) {
		ArrayList<Integer> preList = new ArrayList<Integer>();

		if (root == null) {
			return preList;
		}

		preList.add(root.val);

		for (Integer i : preorderTraversalRecursive(root.left)) {
			preList.add(i);
		}

		for (Integer i : preorderTraversalRecursive(root.right)) {
			preList.add(i);
		}

		return preList;
	}

	/**
	 * Binary Tree Postorder Traversal
	 * 
	 * Given a binary tree, return the post-order traversal of its nodes'
	 * values.
	 * 
	 * For example: Given binary tree {1,#,2,3}, return [3, 2, 1]
	 * 
	 * Note: Recursive solution is trivial, could you do it iteratively?
	 */
	public ArrayList<Integer> postorderTraversalIterative(TreeNode root) {
		ArrayList<Integer> postList = new ArrayList<Integer>();

		// the while loop below don't handle null, so null won't be pushed in s
		if (root == null) {
			return postList;
		}

		// initial the stack which will store the TreeNodes postorderly
		Stack<TreeNode> s = new Stack<TreeNode>();
		s.push(root);

		TreeNode n;
		while (!s.isEmpty()) {
			n = s.pop();

			if (n.right == null && n.left == null) {
				postList.add(n.val);
			} else {
				s.push(new TreeNode(n.val));

				if (n.right != null) {
					s.push(n.right);
				}

				if (n.left != null) {
					s.push(n.left);
				}
			}
		}

		return postList;
	}

	public ArrayList<Integer> postorderTraversalRecursive(TreeNode root) {
		ArrayList<Integer> postList = new ArrayList<Integer>();

		if (root == null) {
			return postList;
		}

		for (Integer i : postorderTraversalRecursive(root.left)) {
			postList.add(i);
		}

		for (Integer i : postorderTraversalRecursive(root.right)) {
			postList.add(i);
		}

		postList.add(root.val);

		return postList;
	}

	/**
	 * Binary Tree Level Order Traversal
	 * 
	 * Given a binary tree, return the level order traversal of its nodes'
	 * values. (ie, from left to right, level by level).
	 */
	public ArrayList<ArrayList<Integer>> levelOrder(TreeNode root) {
		// construct the level order list
		ArrayList<ArrayList<Integer>> lol = new ArrayList<ArrayList<Integer>>();

		if (root == null) {
			return lol;
		}

		// store the TreeNode in sequence of visit
		Queue<TreeNode> queue = new LinkedList<TreeNode>();
		queue.add(root);
		queue.add(null); // use null to separate different level

		while (!queue.isEmpty()) {
			// store integer of each level
			ArrayList<Integer> level = new ArrayList<Integer>();

			while (queue.peek() != null) {
				TreeNode n = queue.poll();

				if (n.left != null) {
					queue.add(n.left);
				}
				if (n.right != null) {
					queue.add(n.right);
				}

				level.add(n.val);
			}

			queue.poll();
			lol.add(level);

			if (!queue.isEmpty()) {
				queue.add(null);
			}
		}

		return lol;
	}

	/**
	 * Binary Tree Level Order Traversal II
	 * 
	 * Given a binary tree, return the bottom-up level order traversal of its
	 * nodes' values. (ie, from left to right, level by level from leaf to
	 * root).
	 */
	public ArrayList<ArrayList<Integer>> levelOrderBottom(TreeNode root) {
		ArrayList<ArrayList<Integer>> lol = levelOrder(root);
		ArrayList<ArrayList<Integer>> lob = new ArrayList<ArrayList<Integer>>(
				lol.size());

		for (int i = lol.size() - 1; i >= 0; i--) {
			lob.add(lol.get(i));
		}

		return lob;
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
	 * Remove Duplicates from Sorted Array
	 * 
	 * Given a sorted array, remove the duplicates in place such that each
	 * element appear only once and return the new length.
	 * 
	 * Do not allocate extra space for another array, you must do this in place
	 * with constant memory.
	 * 
	 * For example, Given input array A = [1,1,2],
	 * 
	 * Your function should return length = 2, and A is now [1,2].
	 * 
	 * Simple optimization: instead of left shift one step every time, find all
	 * the duplicates then shift. This will be efficient if a lot same
	 * duplicates appear
	 */
	public int removeDuplicates(int[] A) {
		int size = A.length;
		
		// case with no duplicates
		if (size < 2) {
			return size;
		}
		
		// once a duplicate is found, shift all following numbers left
		int dup = A[0];
		for (int i = 1; i < size; i++) {
			if (A[i] == dup) {
				int end; // end index of the same duplicate
				for (end = i; end + 1 < size && A[end + 1] == dup;) {
					end++;
				}
				int len = end - i + 1; // length of this set of duplicates
				
				// left shift the part at the right of the set of duplicates
				for (int j = i; j + len < size; j++) {
					A[j] = A[j + len]; 
				}
				size -= len;
			}
			
			dup = A[i];
		}
		
		return size;
	}

	/**
	 * Remove Duplicates from Sorted Array II
	 * 
	 * Follow up for "Remove Duplicates": What if duplicates are allowed at most
	 * twice?
	 * 
	 * For example, Given sorted array A = [1,1,1,2,2,3], your function should
	 * return length = 5, and A is now [1,1,2,2,3].
	 */
	public int removeDuplicates2(int[] A) {
        int size = A.length;
		
		// case with no duplicates
		if (size < 3) {
			return size;
		}
		
		// once a duplicate is found, shift all following numbers left
		int dup = A[0];
		for (int i = 1; i < size; i++) {
			if (A[i] == dup) {
				int end; // end index of the same duplicate
				for (end = i; end + 1 < size && A[end + 1] == dup;) {
					end++;
				}
				int len = end - i + 1; // length of this set of duplicates
				
				// the only additional code for this follow up problem
				if (len > 1) {
				    len--;
				    i++;
    				// left shift the part at the right of the set of duplicates
    				for (int j = i; j + len < size; j++) {
    					A[j] = A[j + len]; 
    				}
    				size -= len;
				}
			}
			
			// update the duplicate value
			dup = A[i];
		}
		
		return size;
    }
	
	/**
	 * Climbing Stairs
	 * 
	 * You are climbing a stair case. It takes n steps to reach to the top.
	 * 
	 * Each time you can either climb 1 or 2 steps. In how many distinct ways
	 * can you climb to the top?
	 * 
	 * Too simple, it's just like Fibonacci, we can even make it O(logn) or O(1)
	 */
	public int climbStairs(int n) {
		int[] steps = new int[n + 1];

		steps[0] = 1;
		steps[1] = 1;

		for (int i = 2; i <= n; i++) {
			steps[i] = steps[i - 1] + steps[i - 2];
		}

		return steps[n];
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
	 * Symmetric Tree
	 * 
	 * Given a binary tree, check whether it is a mirror of itself (ie,
	 * symmetric around its center).
	 */
	public boolean isSymmetricIterative(TreeNode root) {
		if (root == null) {
			return true;
		}

		Queue<TreeNode> queue = new LinkedList<TreeNode>();
		queue.add(root.left);
		queue.add(root.right);

		while (!queue.isEmpty()) {
			TreeNode m = queue.poll();
			TreeNode n = queue.poll();

			if (m != null && n != null) {
				if (m.val == n.val) {
					queue.add(m.left);
					queue.add(n.right);

					queue.add(m.right);
					queue.add(n.left);
				} else {
					return false;
				}
			} else if (m == null && n == null) {
				continue;
			} else {
				return false;
			}
		}

		return true;
	}

	public boolean isSymmetricRecursive(TreeNode root) {
		if (root == null) {
			return true;
		}

		return isSymmetricRecursive(root.left, root.right);
	}

	public boolean isSymmetricRecursive(TreeNode n1, TreeNode n2) {
		if (n1 == null && n2 == null) {
			return true;
		} else if (n1 == null || n2 == null) {
			return false;
		} else {
			if (n1.val == n2.val) {
				return isSymmetricRecursive(n1.left, n2.right)
						&& isSymmetricRecursive(n1.right, n2.left);
			} else {
				return false;
			}
		}
	}

	/**
	 * Merge Two Sorted Lists
	 * 
	 * Merge two sorted linked lists and return it as a new list. The new list
	 * should be made by splicing together the nodes of the first two lists.
	 */
	public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
		// if one of the list is empty, return the other one
		if (l1 == null) {
			return l2;
		} else if (l2 == null) {
			return l1;
		}

		// find the head node
		ListNode head;
		if (l1.val < l2.val) {
			head = l1;
			l1 = l1.next;
		} else {
			head = l2;
			l2 = l2.next;
		}

		// set a pointer pointing to the last element in merged list
		ListNode current = head;

		// merge the two lists until one of them gets empty
		while (l1 != null && l2 != null) {
			if (l1.val < l2.val) {
				current.next = l1;
				l1 = l1.next;
			} else {
				current.next = l2;
				l2 = l2.next;
			}
			current = current.next;
		}

		// add the remaining elements to the merged list
		if (l1 != null) {
			current.next = l1;
		} else {
			current.next = l2;
		}

		return head;
	}

	/**
	 * Merge Sorted Array
	 * 
	 * Given two sorted integer arrays A and B, merge B into A as one sorted
	 * array.
	 * 
	 * Note: You may assume that A has enough space to hold additional elements
	 * from B. The number of elements initialized in A and B are m and n
	 * respectively.
	 */
	public void mergeTwoArrays(int A[], int m, int B[], int n) {
		// index to insert number
		int index = m + n - 1;

		// move the largest one of remaining elements to the end of A, before
		// last moved one
		m--;
		n--;
		while (n >= 0 && m >= 0) {
			if (A[m] > B[n]) {
				A[index] = A[m];
				m--;
			} else {
				A[index] = B[n];
				n--;
			}
			index--;
		}

		// if n is not empty, copy remaining elements to the beginning of A
		if (n >= 0) {
			for (int i = 0; i <= n; i++) {
				A[i] = B[i];
			}
		}
	}

	/**
	 * Swap Nodes in Pairs
	 * 
	 * Given a linked list, swap every two adjacent nodes and return its head.
	 * 
	 * For example, Given 1->2->3->4, you should return the list as 2->1->4->3.
	 * 
	 * Your algorithm should use only constant space. You may not modify the
	 * values in the list, only nodes itself can be changed.
	 */
	public ListNode swapPairs(ListNode head) {
		ListNode prev, first, second, next;
		if (head != null && head.next != null) {
			first = head;
			second = first.next;
			next = second.next;

			head = second;
			second.next = first;
			first.next = next;
			prev = first;
		} else {
			return head;
		}

		while (prev.next != null && prev.next.next != null) {
			first = prev.next;
			second = first.next;
			next = second.next;

			prev.next = second;
			second.next = first;
			first.next = next;
			prev = first;
		}

		return head;
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
	 * Balanced Binary Tree
	 * 
	 * Given a binary tree, determine if it is height-balanced.
	 * 
	 * For this problem, a height-balanced binary tree is defined as a binary
	 * tree in which the depth of the two subtrees of every node never differ by
	 * more than 1.
	 */
	public boolean isBalanced(TreeNode root) {
        return heightWithBalanceCheck(root) != -1;
    }
	
	public int heightWithBalanceCheck(TreeNode root) {
		if (root == null) {
			return 0;
		}
		
		int leftHeight = heightWithBalanceCheck(root.left);
		if (leftHeight == -1) {
			return -1;
		}
		
		int rightHeight = heightWithBalanceCheck(root.right);
		if (rightHeight == -1) {
			return -1;
		}
		
		if (Math.abs(leftHeight - rightHeight) > 1) {
			return -1;
		}

		return 1 + Math.max(leftHeight, rightHeight);
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
	 * Permutations
	 * 
	 * Given a collection of numbers, return all possible permutations.
	 * 
	 * For example, [1,2,3] have the following permutations: [1,2,3], [1,3,2],
	 * [2,1,3], [2,3,1], [3,1,2], and [3,2,1].
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
        ArrayList<String> list = new ArrayList<String>();
        
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < n; i++) {
            sb.append("(");
        }
        for (int i = 0; i < n; i++) {
            sb.append(")");
        }
        list.add(sb.toString());
        
        sb.
    }
	
	public void test() {
		// int[] A = { 1, 2, 3, 3, 4, 4, 5 };
		TreeNode node = new TreeNode(1);
		node.left = new TreeNode(2);
		System.out.println(levelOrder(node));
	}

	public static void main(String[] args) {
		Problems m = new Problems();
		m.test();
	}

}
