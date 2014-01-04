package leet_code;

import java.util.ArrayList;
import java.util.Arrays;
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
	 * Convert Sorted List to Binary Search Tree
	 * 
	 * Given a singly linked list where elements are sorted in ascending order,
	 * convert it to a height balanced BST.
	 */
	public TreeNode sortedListToBST(ListNode head) {
        if (head == null) {
            return null;
        }
        
        // convert the list to ArrayList
        ArrayList<Integer> list = new ArrayList<Integer>();
        while (head != null) {
            list.add(head.val);
            head = head.next;
        }
        
        // recursively construct the binary search tree
        return sortedListToBST(list, 0, list.size() - 1);
    }
    
    public TreeNode sortedListToBST(ArrayList<Integer> list, int start, int end) {
        // only 1 element
        if (start == end) {
            return new TreeNode(list.get(start));
        }
        
        // only 2 elements
        if (end - start == 1) {
            TreeNode root = new TreeNode(list.get(end));
            root.left = new TreeNode(list.get(start));
            return root;
        }
        
        // 3 or more elements
        int mid = (start + end) / 2;
        TreeNode root = new TreeNode(list.get(mid));
        root.left = sortedListToBST(list, start, mid - 1);
        root.right = sortedListToBST(list, mid + 1, end);
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
	 * Permutations II
	 * 
	 * Given a collection of numbers that might contain duplicates, return all
	 * possible unique permutations.
	 * 
	 * For example, [1,1,2] have the following unique permutations: [1,1,2],
	 * [1,2,1], and [2,1,1].
	 */
	public ArrayList<ArrayList<Integer>> permuteUnique(int[] num) {
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
	 * Minimum Path Sum
	 * 
	 * Given a m x n grid filled with non-negative numbers, find a path from top
	 * left to bottom right which minimizes the sum of all numbers along its
	 * path.
	 * 
	 * Note: You can only move either down or right at any point in time.
	 */
	public int minPathSum(int[][] grid) {
        int length = grid.length;
        int width = grid[0].length;
        
        int[][] minSum = new int[length][width];
        
        // initialize the first row
        minSum[0][0] = grid[0][0];
        for (int i = 1; i < width; i++) {
            minSum[0][i] = minSum[0][i-1] + grid[0][i];
        }
        
        // initialize the first column
        for (int i = 1; i < length; i++) {
            minSum[i][0] = minSum[i-1][0] + grid[i][0];
        }
        
        // fill all blank left
        for (int i = 1; i < length; i++) {
            for (int j = 1; j < width; j++) {
                minSum[i][j] = Math.min(minSum[i-1][j], minSum[i][j-1]) + grid[i][j];
            }
        }
        
        return minSum[length-1][width-1];
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
	 * Search a 2D Matrix
	 * 
	 * Write an efficient algorithm that searches for a value in an m x n
	 * matrix. This matrix has the following properties:
	 * 
	 * Integers in each row are sorted from left to right. The first integer of
	 * each row is greater than the last integer of the previous row.
	 */
	public boolean searchMatrix(int[][] matrix, int target) {
		int length = matrix.length;
        int width = matrix[0].length;
        
        int row = binarySearchRow(matrix, target, 0, length - 1);
        if (row == -1) {
            return false;
        }
        
        return binarySearchArray(target, matrix[row], 0, width - 1);
    }
    
    public int binarySearchRow(int[][] matrix, int target, int left, int right) {
        if (left == right) {
            if (matrix[left][0] <= target || matrix[left][matrix[0].length-1] >= target) {
                return left;
            } else {
                return -1;
            }
        } else if (right - left == 1) {
            if (matrix[right][0] <= target) {
                return binarySearchRow(matrix, target, right, right);
            } else {
                return binarySearchRow(matrix, target, left, left);
            }
        } else {
            int mid = (left + right) / 2;
            
            if (matrix[mid][0] > target) {
                return binarySearchRow(matrix, target, left, mid - 1);
            } else if (matrix[mid][matrix[0].length-1] < target) {
                return binarySearchRow(matrix, target, mid + 1, right);
            } else {
                return mid;
            }
        }
    }
    
    public boolean binarySearchArray(int target, int[] array, int left, int right) {
        if (left == right) {
            return target == array[left];
        } else if (right - left == 1) {
            return target == array[left] || target == array[right];
        } else {
            int mid = (left + right) / 2;
            if (target < array[mid]) {
                return binarySearchArray(target, array, left, mid - 1);
            } else if (target > array[mid]) {
                return binarySearchArray(target, array, mid + 1, right);
            } else {
                return true;
            }
        }
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
	 * Combinations
	 * 
	 * Given two integers n and k, return all possible combinations of k numbers
	 * out of 1 ... n.
	 */
	public ArrayList<ArrayList<Integer>> combine(int n, int k) {
        ArrayList<ArrayList<Integer>> combinations = new ArrayList<ArrayList<Integer>>();
        
        if (n == 0 || k == 0 || n < k) {
            return combinations;
        }
        
        combinations.add(new ArrayList<Integer>());
        
        for (int i = 1; i <= n; i++) {
            int len = combinations.size();
            System.out.println(i + " " +combinations);
            // add new lists that contain i for lists that are not full
            for (int j = 0; j < len; j++) {
                ArrayList<Integer> oldList = combinations.get(j);
                
                // list that not full
                if (oldList.size() < k) {
                    // list that must contain all last integers
                    if (k - oldList.size() == n - i + 1) {
                        // add all last integers to the list
                        for (int num = i; num <= n; num++) {
                            oldList.add(num);
                        }
                    } else {
                        // copy the old list and add i to it,
                        // then add the new list to the combinations
                        ArrayList<Integer> newList = new ArrayList<Integer>(oldList);
                        newList.add(i);
                        combinations.add(newList);
                    }
                }
            }
        }
        
        return combinations;
    }
	
	/**
	 * Remove Nth Node From End of List
	 * 
	 * Given a linked list, remove the nth node from the end of list and return
	 * its head.
	 */
	public ListNode removeNthFromEnd(ListNode head, int n) {
        // two runner method
        ListNode slow = head;
        ListNode fast = head;
        
        // move the fast runner to forward by n steps
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        
        // the length of the list is n
        if (fast == null) {
            return head.next;
        }
        
        // move fast and slow simultaneously until fast hit the end
        while (fast.next != null) {
            slow = slow.next;
            fast = fast.next;
        }
        
        // delete the node next to slow
        slow.next = slow.next.next;
        
        return head;
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
	 * Search in Rotated Sorted Array
	 * 
	 * Suppose a sorted array is rotated at some pivot unknown to you
	 * beforehand.
	 * 
	 * (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
	 * 
	 * You are given a target value to search. If found in the array return its
	 * index, otherwise return -1.
	 * 
	 * You may assume no duplicate exists in the array.
	 */
	public int searchRotated(int[] A, int target) {
        return binarySearchRotated(A, 0, A.length - 1, target);
    }
    
    public int binarySearchRotated(int[] A, int left, int right, int target) {
        if (left == right) {
            if (A[left] == target) {
                return left;
            } else {
                return -1;
            }
            
        } else if (right - left == 1) {
            if (A[left] == target) {
                return left;
            } else if (A[right] == target) {
                return right;
            } else {
                return -1;
            }
            
        } else {
            int mid = (left + right) / 2;
            
            if (A[mid] == target) {
                return mid;
            
            } else if ((A[mid] < target && (A[right] >= target || A[right] < A[mid])) 
            		|| (A[mid] > target && A[left] > target && A[left] < A[mid])) {
                return binarySearchRotated(A, mid + 1, right, target);
            
            } else {
                return binarySearchRotated(A, left, mid - 1, target);
            }
        }
    }
	
	/**
	 * Search in Rotated Sorted Array II
	 * 
	 * Follow up for "Search in Rotated Sorted Array": What if duplicates are
	 * allowed?
	 * 
	 * Would this affect the run-time complexity? How and why?
	 * 
	 * Write a function to determine if a given target is in the array.
	 * 
	 * O(logN) ~ O(n), depends on number of duplicates.
	 * 
	 * This solutions is so concise and beautiful.
	 */
	public boolean searchRotatedWithDup(int[] A, int target) {
		int left = 0;
        int right = A.length - 1;
        
        while (left <= right) {
            int mid = (left + right) / 2;
            
            if (A[mid] == target) {
                return true; // or return index according to requirement
            }
            
            if (A[left] < A[mid]) { // left part is sorted
                if (A[left] <= target && A[mid] >= target) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
                
            } else if (A[left] > A[mid]) { // right part is sorted
                if (A[mid] <= target && A[right] >= target) {
                    left = mid;
                } else {
                    right = mid - 1;
                }
                
            } else {
                left++;
            }
        }
        
        return false;
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

		int y = 0;
		while (x != 0) {
			// x == y means the original x is symmetrical
			if (x == y) {
				return true;
			}

			// update y
			y = y * 10 + x % 10;

			// deal with odd digits number
			if (x == y) {
				return true;
			}

			// update x
			x /= 10;
		}

		return false;
	}
	
	/**
	 * Minimum Depth of Binary Tree
	 * 
	 * Given a binary tree, find its minimum depth.
	 * 
	 * The minimum depth is the number of nodes along the shortest path from the
	 * root node down to the nearest leaf node.
	 */
	public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        
        int depth = 1;
        
        // construct queue with first level which only contains root
        Queue<LinkedList<TreeNode>> queues = new LinkedList<LinkedList<TreeNode>>();
        LinkedList<TreeNode> firstLevel = new LinkedList<TreeNode>();
        firstLevel.add(root);
        queues.add(firstLevel);
        
        while (!queues.isEmpty()) {
        	LinkedList<TreeNode> queue = queues.poll();
            
        	// construct list of next level
        	LinkedList<TreeNode> nextLevel = new LinkedList<TreeNode>();
            for (TreeNode n : queue) {
                if (n.left == null && n.right == null) { // leaf node
                    return depth;
                    
                } else {
                    if (n.left != null) {
                        nextLevel.add(n.left);
                    }
                    
                    if (n.right != null) {
                        nextLevel.add(n.right);
                    }
                }
            }
            
            depth++;
            queues.add(nextLevel);
        }
        
        return depth;
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
	 * Subsets
	 * 
	 * Given a set of distinct integers, S, return all possible subsets.
	 * 
	 * Note: Elements in a subset must be in non-descending order. The solution
	 * set must not contain duplicate subsets.
	 */
    public ArrayList<ArrayList<Integer>> subsets(int[] S) {
        ArrayList<ArrayList<Integer>> sets = new ArrayList<ArrayList<Integer>>();
        
        // empty S
        if (S.length == 0) {
            return sets;
        } else {
            Arrays.sort(S);
        }
        
        // add initial empty set
        sets.add(new ArrayList<Integer>());
        
        // add one element to subsets at one time
        for (int i = 0; i < S.length; i++) {
            // add the new element to old subsets to construct new ones
            ArrayList<ArrayList<Integer>> newSubsets = new ArrayList<ArrayList<Integer>>();
            for (ArrayList<Integer> subset : sets) {
                ArrayList<Integer> newSubset = new ArrayList<Integer>(subset);
                newSubset.add(S[i]);
                newSubsets.add(newSubset);
            }
            sets.addAll(newSubsets);
        }
        
        return sets;
    }
    
    /**
     * Subsets II
     * 
     * 
     */
    public ArrayList<ArrayList<Integer>> subsetsWithDup(int[] num) {
        ArrayList<ArrayList<Integer>> sets = new ArrayList<ArrayList<Integer>>();
        
        // empty Array num
        if (num.length == 0) {
            return sets;
        } else {
            Arrays.sort(num);
        }
        
        // add initial empty set
        sets.add(new ArrayList<Integer>());
        
        // add one element to subsets at one time
        for (int i = 0; i < num.length; i++) {
            int dup = 1;
            while (i + 1 < num.length && num[i] == num[i + 1]) {
                i++;
                dup++;
            }
            
            // add the new element to old subsets to construct new ones
            ArrayList<ArrayList<Integer>> oldSubsets = sets;
            while (dup > 0) {
                ArrayList<ArrayList<Integer>> newSubsets = new ArrayList<ArrayList<Integer>>();
                
                for (ArrayList<Integer> subset : oldSubsets) {
                    // add new subset only if the old one contains the new number
                    ArrayList<Integer> newSubset = new ArrayList<Integer>(subset);
                    newSubset.add(num[i]);
                    newSubsets.add(newSubset);
                }
                
                // add new generated subsets to result
                sets.addAll(newSubsets);
                // update oldSubsets
                oldSubsets = newSubsets;
                
                dup--;
            }
        }
        
        return sets;
    }
    
	/**
	 * Longest Common Prefix
	 * 
	 * Write a function to find the longest common prefix string amongst an
	 * array of strings.
	 */
    public String longestCommonPrefix(String[] strs) {
        if (strs.length == 0) {
            return "";
        }
        
        // use StringBuffer to store every temporary prefix
        StringBuffer prefix = new StringBuffer();
        prefix.append(strs[0]);
        
        for (int i = 1; i < strs.length; i++) {
            String s = strs[i];
            
            // trim the size of the prefix
            if (s.length() < prefix.length()) {
                prefix.delete(s.length(), prefix.length());
            }
            
            // compare the old prefix and new string
            for (int j = 0; j < prefix.length(); j++) {
                if (prefix.charAt(j) != s.charAt(j)) {
                    prefix.delete(j, prefix.length());
                    break;
                }
            }
            
            // prefix become empty string means no new prefix any more
            if (prefix.length() == 0) {
                break;
            }
        }
        
        return prefix.toString();
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
	 * 
	 * For example: A = [2,3,1,1,4], return true. A = [3,2,1,0,4], return false.
	 */
    public boolean canJump(int[] A) {
        // farthest distance can be reach
        int distance = 0;
        
        // traverse A to update the distance
        for (int i = 0; i < A.length && i <= distance; i++) {
            distance = Math.max(distance, i + A[i]);
        }
        
        return distance >= A.length - 1;
    }
    
    /**
	 * Jump Game II
	 * 
	 * Given an array of non-negative integers, you are initially positioned at
	 * the first index of the array.
	 * 
	 * Each element in the array represents your maximum jump length at that
	 * position.
	 * 
	 * Your goal is to reach the last index in the minimum number of jumps.
	 * 
	 * For example: Given array A = [2,3,1,1,4]. The minimum number of jumps to
	 * reach the last index is 2. (Jump 1 step from index 0 to 1, then 3 steps
	 * to the last index.)
	 */
    public int jump(int[] A) {
        // store the shortest number of step to every position
        int[] step = new int[A.length];
        
        // initial step
        step[0] = 0;
        for (int i = 1; i < step.length; i++) {
            step[i] = Integer.MAX_VALUE;
        }
        
        // update step base on A
        for (int j = 1; j <= A[0] && j < step.length; j++) {
            step[j] = Math.min(step[j], step[0] + 1);
        }
        
        for (int i = 1; i < A.length; i++) {
            if (A[i] >= A[i - 1]) {
                for (int j = i + 1; j <= i + A[i] && j < step.length; j++) {
                    step[j] = Math.min(step[j], step[i] + 1);
                }
            }
        }
        
        return step[step.length - 1];
    }
    
    /**
	 * Search for a Range
	 * 
	 * Given a sorted array of integers, find the starting and ending position
	 * of a given target value.
	 * 
	 * Your algorithm's runtime complexity must be in the order of O(log n).
	 * 
	 * If the target is not found in the array, return [-1, -1].
	 * 
	 * For example, Given [5, 7, 7, 8, 8, 10] and target value 8, return [3, 4].
	 */
    public int[] searchRange(int[] A, int target) {
        int[] range = new int[2];
        
        // empty A
        if (A.length == 0) {
            range[0] = -1;
            range[1] = -1;
            return range;
        }

        range[0] = searchStart(A, target, 0, A.length - 1);
        range[1] = range[0] == -1 ? -1 : searchEnd(A, target, range[0], A.length - 1);
        
        return range;
    }

    public int searchStart(int[] A, int target, int left, int right) {
        // base case
        if (left == right) {
            if (A[left] == target) {
                return left;
            } else {
                return -1;
            }
        }
        
        // recursive search
        int mid = (left + right) / 2; // tend to choose left mid
        if (A[mid] >= target) {
            return searchStart(A, target, left, mid);
        } else {
            return searchStart(A, target, mid + 1, right);
        }
    }
    
    public int searchEnd(int[] A, int target, int left, int right) {
        // base case
        if (left == right) {
            if (A[left] == target) {
                return left;
            } else {
                return -1;
            }
        }
        
        // recursive search
        int mid = (left + right + 1) / 2; // tend to choose right mid
        if (A[mid] <= target) {
            return searchEnd(A, target, mid, right);
        } else {
            return searchEnd(A, target, left, mid - 1);
        }
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
    
    /**
	 * Reverse Linked List II
	 * 
	 * Reverse a linked list from position m to n. Do it in-place and in
	 * one-pass.
	 * 
	 * For example: Given 1->2->3->4->5->NULL, m = 2 and n = 4, return
	 * 1->4->3->2->5->NULL.
	 * 
	 * Note: m, n satisfy the following condition: 1 ¡Ü m ¡Ü n ¡Ü length of list.
	 */
    public ListNode reverseBetween(ListNode head, int m, int n) {
        if (head == null) {
            return null;
        } else if (m == n) {
            return head;
        }
        
        // dummy head
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        head = dummy;
        
        // find the node before the first one we want to reverse
        ListNode left = head;
        for (int i = 0; i < m - 1; i++) {
            left = left.next;
        }
        
        // mark the last one in the reversed part
        ListNode tail = left.next;
        
        // reverse the node from m to n
        ListNode prev = tail;
        ListNode current = tail.next;
        for (int i = 0; i < n - m - 1; i++) {
            ListNode next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }
        
        // connect the reversed part with its left and right parts
        tail.next = current.next;
        current.next = prev;
        left.next = current;
        
        return head.next;
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
    
	public void test() {
		// int[] A = { 1, 2, 3, 3, 4, 4, 5 };
		// int[][] matrix = {{0,0,0,5},{4,3,1,4},{0,1,1,4},{1,2,1,3},{0,0,1,1}};
		char[][] board = {{'.','8','7','6','5','4','3','2','1'},{'2','.','.','.','.','.','.','.','.'},{'3','.','.','.','.','.','.','.','.'},{'4','.','.','.','.','.','.','.','.'},{'5','.','.','.','.','.','.','.','.'},{'6','.','.','.','.','.','.','.','.'},{'7','.','.','.','.','.','.','.','.'},{'8','.','.','.','.','.','.','.','.'},{'9','.','.','.','.','.','.','.','.'}};

		System.out.println(isValidSudoku(board));		
	}

	public static void main(String[] args) {
		Problems m = new Problems();
		m.test();
	}

}
