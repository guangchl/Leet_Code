package all;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.PriorityQueue;
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
	 * Remove Duplicates from Sorted List
	 * 
	 * Given a sorted linked list, delete all duplicates such that each element
	 * appear only once.
	 * 
	 * For example: Given 1->1->2, return 1->2. Given 1->1->2->3->3, return
	 * 1->2->3.
	 */
	public ListNode deleteDuplicates(ListNode head) {
		if (head == null) {
			return head;
		}
		
		ListNode current = head;
		while (current.next != null) {
			if (current.next.val == current.val) {
				current.next = current.next.next;
			} else {
				current = current.next;
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
		if (head == null || head.next == null) {
			return head;
		}
		
		// dummy node
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		head = dummy;
		
		// check each different number
		while (head.next != null && head.next.next != null) {
		    if (head.next.val == head.next.next.val) { // duplicate found
		    	int val = head.next.val;
		        head.next = head.next.next.next;
		        while (head.next != null && head.next.val == val) {
    		        head.next = head.next.next;
    		    }
		    } else {
		        head = head.next;
		    }
		}
		
		return dummy.next;
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
	 * Binary Tree Zigzag Level Order Traversal
	 * 
	 * Given a binary tree, return the zigzag level order traversal of its
	 * nodes' values. (ie, from left to right, then right to left for the next
	 * level and alternate between).
	 */
	public ArrayList<ArrayList<Integer>> zigzagLevelOrder(TreeNode root) {
        ArrayList<ArrayList<Integer>> levels = new ArrayList<ArrayList<Integer>>();
        
        if (root == null) {
            return levels;
        }
        
        ArrayList<TreeNode> levelNodes = new ArrayList<TreeNode>();
        levelNodes.add(root);
        
        // indicate the node should be add from right to left
        boolean leftDirection = false;

        while (!levelNodes.isEmpty()) {
            ArrayList<TreeNode> newLevelNodes = new ArrayList<TreeNode>();
            ArrayList<Integer> level = new ArrayList<Integer>();
            
            if (leftDirection) {
                for (int i = levelNodes.size() - 1; i >= 0; i--) {
                    level.add(levelNodes.get(i).val);
                }
            } else {
                for (TreeNode n : levelNodes) {
                    level.add(n.val);
                }
            }
            
            for (TreeNode n : levelNodes) {
                if (n.left != null) {
                    newLevelNodes.add(n.left);
                }
                
                if (n.right != null) {
                    newLevelNodes.add(n.right);
                }
            }

            levels.add(level);
            leftDirection = !leftDirection;
            levelNodes = newLevelNodes;
        }
        
        return levels;
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
    
    public static ListNode pointer = null;

	public TreeNode sortedListToBST2(ListNode head) {
		pointer = head;
		int n = 0;
		while (pointer != null) {
			pointer = pointer.next;
			n++;
		}
		pointer = head;
		return create(0, n - 1);
	}

	private TreeNode create(int start, int end) {
		if (pointer == null)
			return null;

		if (start > end)
			return null;
		int mid = (end - start) / 2 + start;
		TreeNode left = create(start, mid - 1);
		TreeNode root = new TreeNode(pointer.val);
		root.left = left;

		pointer = pointer.next;

		root.right = create(mid + 1, end);
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
	
	public ArrayList<ArrayList<Integer>> permuteTemp(int[] num) {
		Arrays.sort(num);
		ArrayList<ArrayList<Integer>> results = new ArrayList<ArrayList<Integer>>();
		ArrayList<Integer> path = new ArrayList<Integer>();
		permuteHelper(results, path, num/* ? */);
		return results;
	}
	
	private void permuteHelper(ArrayList<ArrayList<Integer>> results, ArrayList<Integer> path, int[] num) {
		/* ? */
		for (int i = /* ? */ 0; i < num.length; i++) {
			/* ? skip some cases ? */
			if (path.contains(num[i])) {
				continue;
			}
			path.add(num[i]);
			permuteHelper(results, path, num);
			path.remove(path.size() - 1);
		}
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
		ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
		ArrayList<Integer> tmp = new ArrayList<Integer>();
		Arrays.sort(S);
		res.add(tmp);
		dfs(res, tmp, S, 0);
		return res;
	}

	public void dfs(ArrayList<ArrayList<Integer>> res, ArrayList<Integer> tmp,
			int[] S, int pos) {
		for (int i = pos; i <= S.length - 1; i++) {
			tmp.add(S[i]);
			res.add(new ArrayList<Integer>(tmp));
			dfs(res, tmp, S, i + 1);
			tmp.remove(tmp.size() - 1);
		}
    }
    
    /**
     * Subsets II
     */
    public ArrayList<ArrayList<Integer>> subsetsWithDup(int[] num) {
		ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
		ArrayList<Integer> tmp = new ArrayList<Integer>();
		Arrays.sort(num);
		res.add(tmp);
		dfsWithDup(res, tmp, num, 0);
		return res;
	}

	public void dfsWithDup(ArrayList<ArrayList<Integer>> res, ArrayList<Integer> tmp,
			int[] num, int pos) {
		for (int i = pos; i <= num.length - 1; i++) {
			tmp.add(num[i]);
			res.add(new ArrayList<Integer>(tmp));
			dfsWithDup(res, tmp, num, i + 1);
			tmp.remove(tmp.size() - 1);
			// the only one line difference
			while (i < num.length - 1 && num[i] == num[i + 1]) i++;
		}
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
	
    public int jump2(int[] A) {
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
     * Reverse Nodes in k-Group
     */
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode prev = dummy;
        
        ListNode current = prev.next;
        while (current != null) {
            int count = k - 1;
            while (count > 0 && current != null) {
                current = current.next;
                count--;
            }
            
            if (count == 0 && current != null) {
                // break;
                ListNode next = current.next;
                current.next = null;
                ListNode tail = reverseToTail(prev.next);
                // reconnect
                prev.next = current;
                tail.next = next;
                // update
                prev = tail;
                current = next;
            } else {
                return dummy.next;
            }
        }
        
        return dummy.next;
    }
    
    public ListNode reverseToTail(ListNode head) {
        ListNode tail = head;
        ListNode current = head.next;
        head.next = null;

        while (current != null) {
            ListNode next = current.next;
            current.next = head;
            head = current;
            current = next;
        }
        return tail;
    }
    
    /**
     * Construct Binary Tree from Preorder and Inorder Traversal
     * 
	 * Given preorder and inorder traversal of a tree, construct the binary
	 * tree.
	 * 
	 * Note: You may assume that duplicates do not exist in the tree.
	 */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int len = preorder.length;
        if (len == 0 || len != inorder.length) {
            return null;
        }
        
        // root is the first in preorder
        int rootVal = preorder[0];
        TreeNode root = new TreeNode(rootVal);
        
        // only one element
        if (len == 1) {
            return root;
        }
        
        // find the root index in inorder
        int i;
        for (i = 0; i < len; i++) {
            if (inorder[i] == rootVal) {
                break;
            }
        }
        
        // left subtree
        if (i > 0) {
            root.left = buildTree(Arrays.copyOfRange(preorder, 1, i + 1), Arrays.copyOfRange(inorder, 0, i));
        }
        
        // right subtree
        if (i < len - 1) {
            root.right = buildTree(Arrays.copyOfRange(preorder, i + 1, len), Arrays.copyOfRange(inorder, i + 1, len));
        }
        
        return root;
    }
    
    /**
	 * Construct Binary Tree from Inorder and Postorder Traversal
	 * 
	 * Given inorder and postorder traversal of a tree, construct the binary
	 * tree.
	 * 
	 * Note: You may assume that duplicates do not exist in the tree.
	 */
    public TreeNode buildTree2(int[] inorder, int[] postorder) {
        int len = postorder.length;
        if (len == 0 || len != inorder.length) {
            return null;
        }
        
        // root is the last in postorder
        int rootVal = postorder[len - 1];
        TreeNode root = new TreeNode(rootVal);
        
        // only one element
        if (len == 1) {
            return root;
        }
        
        // find the root index in inorder
        int i;
        for (i = 0; i < len; i++) {
            if (inorder[i] == rootVal) {
                break;
            }
        }
        
        // left subtree
        if (i > 0) {
            root.left = buildTree2(Arrays.copyOfRange(inorder, 0, i), Arrays.copyOfRange(postorder, 0, i));
        }
        
        // right subtree
        if (i < len - 1) {
            root.right = buildTree2(Arrays.copyOfRange(inorder, i + 1, len), Arrays.copyOfRange(postorder, i, len - 1));
        }
        
        return root;
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
	 * Partition List
	 * 
	 * Given a linked list and a value x, partition it such that all nodes less
	 * than x come before nodes greater than or equal to x.
	 * 
	 * You should preserve the original relative order of the nodes in each of
	 * the two partitions.
	 * 
	 * For example, Given 1->4->3->2->5->2 and x = 3, return 1->2->2->4->3->5.
	 */
    public ListNode partition(ListNode head, int x) {
        if (head == null) {
            return null;
        }
        
        // construct a list store all node greater or equals to x
        ListNode greaterHead = null;
        ListNode greaterTail = null;
        
        // construct a dummy node to be head
        ListNode dummy = new ListNode(x - 1);
        dummy.next = head;
        head = dummy;
        ListNode current = head;

        // find the first node belongs to the greater part
        while (current.next != null && current.next.val < x) {
            current = current.next;
        }
        
        // if no node have val greater or equals to x
        if (current.next == null) {
            return head.next;
        } else { // find the first one, add it to greater list
            greaterHead = current.next;
            greaterTail = current.next;
            current.next = current.next.next;
        }
        
        // traverse remaining nodes, add greater node to greater list
        while (current.next != null) {
            if (current.next.val >= x) {
                greaterTail.next = current.next;
                greaterTail = greaterTail.next;
                current.next = current.next.next;
            } else {
                current = current.next;
            }
        }
        
        // connect the greater list next to the original list
        current.next = greaterHead;
        greaterTail.next = null;
        
        return head.next;
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
	 * Triangle
	 * 
	 * Given a triangle, find the minimum path sum from top to bottom. Each step
	 * you may move to adjacent numbers on the row below.
	 * 
	 * Note: Bonus point if you are able to do this using only O(n) extra space,
	 * where n is the total number of rows in the triangle.
	 */
    public int minimumTotal(ArrayList<ArrayList<Integer>> triangle) {
        int depth = triangle.size();
        int[] minPath = new int[depth];
        
        // start from the last level
        ArrayList<Integer> lastLevel = triangle.get(depth - 1);
        for (int i = 0; i < depth; i++) {
            minPath[i] = lastLevel.get(i);
        }
        
        // calculate the minimum path to each node
        for (int i = depth - 2; i >= 0; i--) {
            ArrayList<Integer> level = triangle.get(i);
            
            for (int j = 0; j < level.size(); j++) {
                minPath[j] = level.get(j) + Math.min(minPath[j], minPath[j + 1]);
            }
        }
        
        return minPath[0];
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
	 * Validate Binary Search Tree
	 * 
	 * Given a binary tree, determine if it is a valid binary search tree (BST).
	 * 
	 * Assume a BST is defined as follows: The left subtree of a node contains
	 * only nodes with keys less than the node's key. The right subtree of a
	 * node contains only nodes with keys greater than the node's key. Both the
	 * left and right subtrees must also be binary search trees.
	 */
    public boolean isValidBST(TreeNode root) {
        return isBST(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
    }
    
    public boolean isBST(TreeNode root, int min, int max) {
        if (root == null) {
            return true;
        }

        return root.val > min && root.val < max && isBST(root.left, min, root.val) && isBST(root.right, root.val, max);
    }
    
    /**
	 * Next Permutation
	 * 
	 * Implement next permutation, which rearranges numbers into the
	 * lexicographically next greater permutation of numbers.
	 * 
	 * If such arrangement is not possible, it must rearrange it as the lowest
	 * possible order (ie, sorted in ascending order).
	 * 
	 * The replacement must be in-place, do not allocate extra memory.
	 * 
	 * Here are some examples. Inputs are in the left-hand column and its
	 * corresponding outputs are in the right-hand column.
	 * 1,2,3 ¡ú 1,3,2
	 * 3,2,1 ¡ú 1,2,3
	 * 1,1,5 ¡ú 1,5,1
	 */
    public void nextPermutation(int[] num) {
        int maxIndex = num.length - 1;
        
        int i;
        for (i = num.length - 2; i >= 0; i--) {
            // find the first position which should be changed
            if (num[i] < num[maxIndex]) {
                // find the minimum among all number larger than num[i]
                int greaterMin = maxIndex;
                for (int j = i + 1; j < num.length; j++) {
                    if (num[j] > num[i] && num [j] < num[greaterMin]) {
                        greaterMin = j;
                    }
                }
                
                // swap the num[i] and the number we found
                int temp = num[i];
                num[i] = num[greaterMin];
                num[greaterMin] = temp;
                break;
            } else {
                maxIndex = i;
            }
        }
        
        Arrays.sort(num, i + 1, num.length);
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
	 * Palindrome Partitioning II
	 * 
	 * Given a string s, partition s such that every substring of the partition
	 * is a palindrome.
	 * 
	 * Return the minimum cuts needed for a palindrome partitioning of s.
	 * 
	 * For example, given s = "aab", Return 1 since the palindrome partitioning
	 * ["aa","b"] could be produced using 1 cut.
	 */
    public int minCut(String s) {
        int len = s.length();
        boolean[][] isPal = new boolean[len][len];
        int[] dp = new int[len + 1];
        
        for(int i = 0; i <= len; i++)
            dp[i] = len - 1 - i;
            
        for(int i = len - 2; i >= 0; i--) {
            for(int j = i; j < len; j++) {
                if(s.charAt(i) == s.charAt(j) && (j <= i+2 || isPal[i+1][j-1])) {
                    isPal[i][j] = true;
                    dp[i] = Math.min(dp[i], dp[j+1] + 1);
                }
            }
        }
        return dp[0];
    }
    
    /**
	 * Word Break
	 * 
	 * Given a string s and a dictionary of words dict, determine if s can be
	 * segmented into a space-separated sequence of one or more dictionary
	 * words.
	 * 
	 * For example, given s = "leetcode", dict = ["leet", "code"].
	 * Return true because "leetcode" can be segmented as "leet code".
	 * 
	 * Bottom up DP seems quick useful
	 */
    public boolean wordBreak(String s, Set<String> dict) {
        int len = s.length();
        boolean[] dp = new boolean[len + 1];
        dp[len] = true;

        for (int i = len - 1; i >= 0; i--) {
            for (int j = i; j < len; j++) {
                String str = s.substring(i,j + 1);
                if (dict.contains(str) && dp[j + 1]) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[0];
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
	 * Rotate List
	 * 
	 * Given a list, rotate the list to the right by k places, where k is
	 * non-negative.
	 * 
	 * For example: Given 1->2->3->4->5->NULL and k = 2, return
	 * 4->5->1->2->3->NULL.
	 */
	public ListNode rotateRight(ListNode head, int n) {
        if (head == null) return null;

        ListNode slow = head;
        ListNode fast = head;
        
        while (n-- > 0) {
            if (fast.next == null) {
                fast = head;
            } else {
                fast = fast.next;
            }
        }
        
        // n = k * length
        if (fast == head) return head;
        
        while (fast.next != null) {
            slow = slow.next;
            fast = fast.next;
        }
        
        fast.next = head;
        head = slow.next;
        slow.next = null;
        
        return head;
    }
	
	/**
	 * LRU Cache
	 * 
	 * Design and implement a data structure for Least Recently Used (LRU)
	 * cache. It should support the following operations: get and set.
	 * 
	 * get(key) - Get the value (will always be positive) of the key if the key
	 * exists in the cache, otherwise return -1. set(key, value) - Set or insert
	 * the value if the key is not already present. When the cache reached its
	 * capacity, it should invalidate the least recently used item before
	 * inserting a new item.
	 */
	public class LRUCache {
		private LinkedHashMap<Integer, Integer> map;
		private int cacheSize;
		private static final float hashTableLoadFactor = .75f;

		public LRUCache(int capacity) {
			this.cacheSize = capacity;
			map = new LinkedHashMap<Integer, Integer>(capacity,
					hashTableLoadFactor, true) {
				private static final long serialVersionUID = 1L;

				protected boolean removeEldestEntry(
						Map.Entry<Integer, Integer> eldest) {
					return size() > LRUCache.this.cacheSize;
				}
			};
		}

		public int get(int key) {
			if (map.containsKey(key))
				return map.get(key);
			return -1;
		}

		public void set(int key, int value) {
			map.put(key, value);
		}
	}
	
	/** This is my solution which cannot promise O(1) for any operation */
	public class MyLRUCache {
	    private HashMap<Integer, Integer> map;
	    private int capacity;
	    
	    private LinkedList<Integer> queue;
	    private int size;
	    
	    public MyLRUCache(int capacity) {
	        map = new HashMap<Integer, Integer>(capacity);
	        this.capacity = capacity;
	        queue = new LinkedList<Integer>();
	        size = 0;
	    }
	    
	    public int get(int key) {
	        if (!map.containsKey(key)) return -1;
	        moveKeyToLast(key);
	        return map.get(key);
	    }
	    
	    public void set(int key, int value) {
	        if (map.containsKey(key)) {
	            map.put(key, value);
	            moveKeyToLast(key);
	        } else {
	            if (size < capacity) {
	                queue.add(key);
	                map.put(key, value);
	                size++;
	            } else { // full
	                // remove old
	                int keyToDelete = queue.poll();
	                map.remove(keyToDelete);
	                
	                // add new
	                queue.add(key);
	                map.put(key, value);
	            }
	        }
	    }
	    
	    private void moveKeyToLast(int key) {
	        for (int i = 0; i < queue.size(); i++) {
	            if (queue.get(i) == key) {
	                queue.remove(i);
	                break;
	            }
	        }
	        queue.add(key);
	    }
	}
	
	/**
	 * This is my optimized solution
	 */
	public class MyLRUCache2 {
	    private HashMap<Integer, DoubleLinkedNode> map;
		private int capacity;
	    
	    private DoubleLinkedNode head;
	    private DoubleLinkedNode tail;
	    private int size;
	    
	    public MyLRUCache2(int capacity) {
	        map = new HashMap<Integer, DoubleLinkedNode>(capacity);
	        this.capacity = capacity;
	        head = null;
	        tail = null;
	        size = 0;
	    }
	    
	    public int get(int key) {
	        if (!map.containsKey(key)) return -1;
	        moveKeyToLast(key);
	        return map.get(key).value;
	    }
	    
	    public void set(int key, int value) {
	        if (map.containsKey(key)) {
	            map.get(key).value = value;
	            moveKeyToLast(key);
	        } else {
	            if (size == capacity) removeFirst(); // remove oldest if full
	            
	            addKeyValue(key, value); // add new
	            size++;
	        }
	    }
	    
	    private void addKeyValue(int key, int value) {
	        DoubleLinkedNode n = new DoubleLinkedNode(key, value);
	        map.put(key, n);
	        
	        if (tail == null) {
	            head = n;
	            tail = n;
	        } else {
	            tail.next = n;
	            n.prev = tail;
	            n.next = null;
	            tail = n;
	        }
	    }
	    
	    private void removeFirst() {
	    	int keyToDelete = head.key;
	    	map.remove(keyToDelete);
	    	
	    	if (capacity == 1) {
	    		head = null;
	    		tail = null;
	    	} else {
	    		head = head.next;
	            head.prev = null;
	    	}

            size--;
	    }
	    
	    private void moveKeyToLast(int key) {
	        DoubleLinkedNode n = map.get(key);
	        
	        if (n == tail) {
	            return;
	        }
	        
	        if (n == head) {
	            head = n.next;
	            head.prev = null;
	        } else {
	            n.prev.next = n.next;
	            n.next.prev = n.prev;
	        }
	        
	        tail.next = n;
	        n.prev = tail;
	        tail = n;
	        tail.next = null;
	    }
	    
	    private class DoubleLinkedNode {
	        int key;
	        int value;
	        
	        DoubleLinkedNode prev;
	        DoubleLinkedNode next;
	        
	        public DoubleLinkedNode(int key, int value) {
	            this.key = key;
	            this.value = value;
	        }
	    }
	}
	
	/**
	 * Add Two Numbers
	 * 
	 * You are given two linked lists representing two non-negative numbers. The
	 * digits are stored in reverse order and each of their nodes contain a
	 * single digit. Add the two numbers and return it as a linked list.
	 * 
	 * Input: (2 -> 4 -> 3) + (5 -> 6 -> 4) Output: 7 -> 0 -> 8
	 */
	public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
		ListNode l = new ListNode(0); // dummy head node
        int sum = 0, carry = 0;
        ListNode current = l;
        
        while (l1 != null && l2 != null) {
            sum = l1.val + l2.val + carry;
            carry = sum / 10;
            current.next = new ListNode(sum % 10);
            current = current.next;
            l1 = l1.next;
            l2 = l2.next;
        }
        
        // done with both l1 and l2
        if (l1 == null && l2 == null) {
            if (carry != 0) current.next = new ListNode(carry);
            return l.next;
        } else  if (l1 != null) { // l1 still need to be added
            current.next = l1;
        } else if (l2 != null) { // l2 still need to be added
            current.next = l2;
        }
        current = current.next;
        
        while (carry != 0 && current.next != null) {
            sum = current.val + carry;
            carry = sum / 10;
            current.val = sum % 10;
            current = current.next;
        }
        sum = current.val + carry;
        carry = sum / 10;
        current.val = sum % 10;
        if (carry != 0) current.next = new ListNode(carry);
        return l.next;
    }
	
	/**
	 * N-Queens
	 * 
	 * Given an integer n, return all distinct solutions to the n-queens puzzle.
	 */
	public ArrayList<String[]> solveNQueens(int n) {
        // IMPORTANT: Please reset any member data you declared, as
        // the same Solution instance will be reused for each test case.
        ArrayList<String[]> res = new ArrayList<String[]>();
        int[] loc = new int[n];
        dfs(res,loc,0,n);
        return res;
    }  
    
    public void dfs(ArrayList<String[]> res, int[] loc, int cur, int n){  
        if(cur == n)   
            printboard(res, loc, n);  
        else{  
            for(int i = 0; i < n; i++){  
                loc[cur] = i;  
                if(isValid(loc, cur))  
                    dfs(res, loc, cur + 1,n);  
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

    /** This is my solution */
    public ArrayList<String[]> solveNQueens2(int n) {
        ArrayList<String[]> finalSolutions = new ArrayList<String[]>();
        Queue<PendingSolution> queue = new LinkedList<PendingSolution>();
        
        // a list of unused Integers
        LinkedList<Integer> allNumbers = new LinkedList<Integer>();
        for (int i = 0; i < n; i++) {
            allNumbers.add(i);
        }
        
        // add first rows for n possible solutions
        for (int i = 0; i < n; i++) {
            // partial solution
            ArrayList<Integer> rows = new ArrayList<Integer>();
            rows.add(i);
            
            // unused number
            LinkedList<Integer> numbers = new LinkedList<Integer>(allNumbers);
            numbers.remove(i);
            
            // add the first level of pending solutions
            PendingSolution ps = new PendingSolution(rows, numbers);
            queue.add(ps);
        }
        
        // find all the possible answer
        while (!queue.isEmpty()) {
            PendingSolution ps = queue.poll();
            if (ps.isComplete()) { // solution found
                finalSolutions.add(ps.toStrings());
            }
            
            ArrayList<Integer> rows = ps.rows;
            LinkedList<Integer> numbers = ps.numbers;

            for (Integer i : numbers) {
                if (!ps.hasDiagonalConflict(i)) {
                    ArrayList<Integer> newRows = new ArrayList<Integer>(rows);
                    newRows.add(i);
                    
                    LinkedList<Integer> newNumbers = new LinkedList<Integer>(numbers);
                    newNumbers.remove(i);
                    
                    PendingSolution newPs = new PendingSolution(newRows, newNumbers);
                    queue.add(newPs);// add pending solution
                }
            }
        }
        System.out.println(finalSolutions);
        return finalSolutions;
    }
    
    public class PendingSolution {
        ArrayList<Integer> rows;
        LinkedList<Integer> numbers;
        
        public PendingSolution(ArrayList<Integer> rows, LinkedList<Integer> numbers) {
            this.rows = rows;
            this.numbers = numbers;
        }
        
        public boolean isComplete() {
            return numbers.size() == 0;
        }
        
        public boolean hasDiagonalConflict(int newNumber) {
            int r = rows.size(); // row number for newNumber
            
            for (int i = 0; i < r; i++) {
                if (Math.abs(r - i) == Math.abs(newNumber - rows.get(i))) {
                    return true;
                }
            }
            return false;
        }
        
        public String[] toStrings() {
            int size = rows.size();
            String[] strings = new String[size];
            
            for (int i = 0; i < strings.length; i++) {
                int number = rows.get(i);
                StringBuffer sb = new StringBuffer();
                
                for (int j = 0; j < number; j++) {
                    sb.append('.');
                }
                sb.append('Q');
                for (int j = number + 1; j < size; j++) {
                    sb.append('.');
                }
                
                strings[i] = sb.toString();
            }
            
            return strings;
        }
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
	 * Word Ladder
	 * 
	 * Given two words (start and end), and a dictionary, find the length of
	 * shortest transformation sequence from start to end, such that:
	 * Only one letter can be changed at a time
	 * Each intermediate word must exist in the dictionary
	 * 
	 * For example, Given:
	 * start = "hit"
	 * end = "cog"
	 * dict = ["hot","dot","dog","lot","log"]
	 * As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" ->
	 * "cog", return its length 5.
	 * 
	 * Note:
	 * Return 0 if there is no such transformation sequence.
	 * All words have the same length.
	 * All words contain only lowercase alphabetic characters.
	 */
    public int ladderLength(String start, String end, HashSet<String> dict) {
    	if (start == null || end == null) {
    		return 0;
    	}
    	
        int len = 2;
        
        // add start to queue, levels are separated by null
        Queue<String> queue = new LinkedList<String>();
        queue.add(start);
        queue.add(null);
        
        // set that mark all visited words
        Set<String> visited = new HashSet<String>();
        if (dict.contains(start))
            visited.add(start);
        
        while (!queue.isEmpty()) {
            String s = queue.poll();
            
            // check level end and search end
            if (s == null) {
                if (queue.size() != 0) {
                	queue.add(null);
                    len++;
                    continue;
                } else {
                	break;
                }
            }

            // change every letter one by one
            for (int i = 0; i < s.length(); i++) {
            	// use StringBuffer to save computation
                StringBuffer sb = new StringBuffer(s);
                
                // replace the character at index i
                for (char j = 'a'; j <= 'z'; j++) {
                    sb.setCharAt(i, j);
                    String newS = sb.toString();
                    if (newS.equals(end))
                        return len;
                    if (dict.contains(newS) && !visited.contains(newS)) {
                        queue.add(newS);
                        visited.add(newS);
                    }
                }
            }
        }
        
        return 0;
    }
    
    /**
	 * Word Ladder II
	 * 
	 * Given two words (start and end), and a dictionary, find all shortest
	 * transformation sequence(s) from start to end, such that:
	 * Only one letter can be changed at a time
	 * Each intermediate word must exist in the dictionary
	 * 
	 * For example, Given:
	 * start = "hit"
	 * end = "cog"
	 * dict = ["hot","dot","dog","lot","log"]
	 * Return
  	 * [
	 * 	["hit","hot","dot","dog","cog"],
	 * 	["hit","hot","lot","log","cog"]
	 * ]
	 * 
	 * Note:
	 * All words have the same length.
	 * All words contain only lowercase alphabetic characters.
	 */
	public ArrayList<ArrayList<String>> findLadders(String start, String end,
			HashSet<String> dict) {
		ArrayList<ArrayList<String>> paths = new ArrayList<ArrayList<String>>();
		if (start == null || end == null || start.length() == 0)
			return paths;
		// maintain a hashmap for visited words
		Map<String, ArrayList<SugeNode>> visited = new HashMap<String, ArrayList<SugeNode>>();
		// BFS to find the minimum sequence length
		getMinLength(start, end, dict, visited);
		// DFS to back trace paths from end to start
		buildPaths(end, start, visited, new LinkedList<String>(), paths);
		return paths;
	}

	/*
	 * Use BFS to find the minimum transformation sequences length from start to
	 * end. Also store parent nodes from previous level for each visited valid
	 * word.
	 */
	private void getMinLength(String start, String end, HashSet<String> dict,
			Map<String, ArrayList<SugeNode>> visited) {
		// maintain a queue for words, depth and previous word during BFS
		Queue<SugeNode> queue = new LinkedList<SugeNode>();
		queue.add(new SugeNode(start, 1));
		// BFS
		dict.add(end);
		while (!queue.isEmpty()) {
			SugeNode node = queue.poll();
			for (int i = 0; i < node.word.length(); ++i) {
				StringBuilder sb = new StringBuilder(node.word);
				char original = sb.charAt(i);
				for (char c = 'a'; c <= 'z'; ++c) {
					if (c == original)
						continue;
					sb.setCharAt(i, c);
					String s = sb.toString();
					if (dict.contains(s) && !s.equals(start)) {
						ArrayList<SugeNode> pres = visited.get(s);
						if (pres == null) {
							// enqueue unvisited word
							queue.add(new SugeNode(s, node.depth + 1));
							pres = new ArrayList<SugeNode>();
							visited.put(s, pres);
							pres.add(node);
						} else if (pres.get(0).depth == node.depth) {
							// parent nodes should be in the same level - to
							// avoid circle in graph
							pres.add(node);
						}
					}
				}
			}
		}

	}

	/* Use DFS to back trace all paths from end to start. */
	private void buildPaths(String s, String start,
			Map<String, ArrayList<SugeNode>> visited, LinkedList<String> path,
			ArrayList<ArrayList<String>> paths) {
		if (visited == null)
			return;
		path.add(0, s);
		if (s.equals(start)) {
			ArrayList<String> p = new ArrayList<String>(path);
			paths.add(p);
		} else {
			ArrayList<SugeNode> pres = visited.get(s);
			if (pres != null) {
				for (SugeNode pre : pres) {
					buildPaths(pre.word, start, visited, path, paths);
				}
			}
		}
		path.remove(0);
	}
    	   
	private class SugeNode {  
		String word;
		int depth;

		public SugeNode(String w, int d) {
			word = w;
			depth = d;
		}
	}
    	 
    /** This is my solution, time limit exceeded */
    public ArrayList<ArrayList<String>> findLadders2(String start, String end, HashSet<String> dict) {
        ArrayList<ArrayList<String>> ladders = new ArrayList<ArrayList<String>>();
        if (start == null || end == null) 
    		return ladders;

        // add start to queue, levels are separated by null
        Queue<ArrayList<String>> queue = new LinkedList<ArrayList<String>>();
        ArrayList<String> begin = new ArrayList<String>();
        begin.add(start);
        queue.add(begin);
        queue.add(null);
        
        // set that mark all visited words
        Set<String> visited = new HashSet<String>();
        if (dict.contains(start))
            visited.add(start);
        // new visited for each level
        Set<String> newVisited = new HashSet<String>();
        
        while (!queue.isEmpty()) {
            ArrayList<String> ladder = queue.poll();
            
            // check level end and search end
            if (ladder == null) {
                if (ladders.size() != 0) // result found
                    return ladders;
                
                if (queue.size() != 0) {
                	visited.addAll(newVisited);
                	visited.clear();
                	queue.add(null);
                    continue;
                } else { // queue is empty
                	break;
                }
            }
            
            int beforeSize = ladder.size();
            String s = ladder.get(beforeSize - 1);

            // change every letter one by one
            for (int i = 0; i < s.length() && ladder.size() == beforeSize; i++) {
            	// use StringBuffer to save computation
                StringBuffer sb = new StringBuffer(s);
                
                // replace the character at index i
                for (char j = 'a'; j <= 'z'; j++) {
                    sb.setCharAt(i, j);
                    String newS = sb.toString();
                    if (newS.equals(end)) {
                        visited.add(end);
                        ladder.add(end);
                        ladders.add(ladder);
                        break;
                    }
                        
                    if (dict.contains(newS) && !visited.contains(newS)) {
                        newVisited.add(newS);
                        ArrayList<String> newLadder = new ArrayList<String>(ladder);
                        newLadder.add(newS);
                        queue.add(newLadder);
                    }
                }
            }
        }
        
        return ladders;
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
     * First Missing Positive
     */
    public int firstMissingPositive(int[] A) {
        int len = A.length;
        if (len == 0) {
            return 1;
        }
        
        for (int i = 0; i < A.length;) {
            int num = A[i];
            if (num > 0 && num <= len && num != i + 1 && num != A[num - 1]) {
                A[i] = A[num - 1];
                A[num - 1] = num;
            } else {
                i++;
            }
        }

        for (int i = 0; i < len; i++) {
            if (A[i] != i + 1) {
                return i + 1;
            }
        }
        
        return len + 1;
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

    /**
	 * Wildcard Matching
	 * 
	 * Implement wildcard pattern matching with support for '?' and '*'.
	 * 
	 * '?' Matches any single character. '*' Matches any sequence of characters
	 * (including the empty sequence).
	 * 
	 * The matching should cover the entire input string (not partial).
	 * 
	 * The function prototype should be: bool isMatch(const char *s, const char
	 * *p)
	 *
     * Some examples:
     * isMatch("aa","a") ¡ú false
     * isMatch("aa","aa") ¡ú true
     * isMatch("aaa","aa") ¡ú false
     * isMatch("aa", "*") ¡ú true
     * isMatch("aa", "a*") ¡ú true
     * isMatch("ab", "?*") ¡ú true
     * isMatch("aab", "c*a*b") ¡ú false
     */
	public boolean isMatch(String s, String p) {
		int is = 0, ip = 0; // search ptrs
		int js = -1; // latest search ptr
		int jp = -1; // latest star ptr

		while (true) {
			if (is == s.length()) {// end of s, check the rest of p
				for (int i = ip; i < p.length(); ++i) {
					if (p.charAt(i) != '*')
						return false;
				}
				return true;
			} else {
				if (ip < p.length()) {
					if (s.charAt(is) == p.charAt(ip) || p.charAt(ip) == '?') {// single
																				// match
						is++;
						ip++;
						continue;
					} else if (p.charAt(ip) == '*') {// star, search next
														// character in p
						js = is;
						jp = ip;
						ip = jp + 1;
						continue;
					}
				}
				// mismatch, check roll back
				if (js >= 0) {// roll back in the star position
					ip = jp + 1;
					js++;
					is = js;
				} else {// hard mismatch
					return false;
				}
			}
		}
	}
	
	/** Solution from internet */
	public boolean isMatch2(String s, String p) {
		if (s == null || p == null)
			return false;

		// calculate count for non-wildcard char
		int count = 0;
		for (Character c : p.toCharArray()) {
			if (c != '*')
				++count;
		}
		// the count should not be larger than that of s
		if (count > s.length())
			return false;

		boolean[] matches = new boolean[s.length() + 1];
		matches[0] = true;
		int pid = 0, firstMatch = 0;
		while (pid < p.length()) {
			// skip duplicate '*'
			if (pid > 0 && p.charAt(pid) == '*' && p.charAt(pid - 1) == '*') {
				++pid;
				continue;
			}

			// if '*', fill up the rest of row
			if (p.charAt(pid) == '*') {
				// fill up the rest of row with true, up to the first match in
				// previous row
				for (int i = firstMatch + 1; i <= s.length(); ++i)
					matches[i] = true;
			} else {
				// fill up backwards:
				// - set to true if match current char and previous diagnal also
				// match
				// - otherwise, set to false
				int match = -1;
				for (int i = s.length(); i > firstMatch; --i) {
					matches[i] = (p.charAt(pid) == s.charAt(i - 1) || p
							.charAt(pid) == '?') && matches[i - 1];
					if (matches[i])
						match = i;
				}
				if (match < 0)
					return false;
				firstMatch = match;
			}

			++pid;
		}

		return matches[s.length()];
	}
    
    /** This is my solution */
    public boolean isMatch3(String s, String p) {
    	int slen = s.length();
        int plen = p.length();
        
        if (slen == 0 && plen == 0) {
            return true;
        } else if (slen == 0) {
            for (int i = 0; i < plen; i++) {
                if (p.charAt(i) != '*')
                    return false;
            }
            return true;
        } else if (plen == 0) {
            return false;
        }
        
        // match[i][j] indicates s from 0 to i matches p from 0 to j
        boolean[][] match = new boolean[slen][plen];
        
        // when p from 0 to end
        if (p.charAt(0) == '*' || p.charAt(0) == '?' || p.charAt(0) == s.charAt(0)) {
            match[0][0] = true;
            for (int i = 1; i < plen; i++) {
                if (p.charAt(i) == '*') {
                    match[0][i] = true;
                } else {
                    break;
                }
            }
        }
        
        // when s from 1 to end
        if (p.charAt(0) == '*') {
        	for (int i = 1; i < slen; i++)
                match[i][0] = true;
        }

        for (int i = 1; i < slen; i++) {
            for (int j = 1; j < plen; j++) {
            	System.out.println();
                switch (p.charAt(j)) {
                	case '?':
                		match[i][j] = match[i - 1][j - 1];
                		break;
                	case '*':
                		match[i][j] = match[i - 1][j] || match[i][j - 1];
                		break;
                	default:
                		match[i][j] = match[i - 1][j - 1] && s.charAt(i) == p.charAt(j);
                }
            }
        }
        
        return match[slen - 1][plen - 1];
    }
    
    /**
     * Insertion Sort List
     * 
     * Sort a linked list using insertion sort.
     */
    public ListNode insertionSortList(ListNode head) {
        if (head == null) return null;
        
        // dummy head
        ListNode dummy = new ListNode(Integer.MIN_VALUE);
        dummy.next = head;
        head = dummy;

        // start from the one next to original head
        ListNode prev = dummy.next;
        ListNode current = prev.next;
        while (current != null) {
            // insert if current is not in order
            if (current.val < prev.val) {
                // move head to last node which smaller than current
                while (head.next.val < current.val) {
                    head = head.next;
                }
                
                // delete current from original position
                prev.next = current.next;
                
                // add current to head.next
                current.next = head.next;
                head.next = current;
                head = dummy;
                
                // update prev and current
                current = prev.next;
            } else {
                // update prev and current
                prev = current;
                current = prev.next;
            }
        }
        
        return dummy.next;
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
     * Reorder List
     * 
     * Given a singly linked list L: L0¡úL1¡ú¡­¡úLn-1¡úLn,
     * reorder it to: L0¡úLn¡úL1¡úLn-1¡úL2¡úLn-2¡ú¡­
     * 
     * You must do this in-place without altering the nodes' values.
     * 
     * For example,
     * Given {1,2,3,4}, reorder it to {1,4,2,3}.
     */
    public void reorderList(ListNode head) {
        if (head == null || head.next == null || head.next.next == null)
            return;
        
        // find the second half
        ListNode slow = head;
        ListNode fast = head.next;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        
        // break the list, reverse the second half list
        ListNode list = slow.next;
        slow.next = null;
        list = reverseSingleLinkedList(list);
        
        // merge the two list
        ListNode current = head;
        while (list != null) {
            ListNode temp = list;
            list = list.next;
            temp.next = current.next;
            current.next = temp;
            current = temp.next;
        }
    }
    
    public ListNode reverseSingleLinkedList(ListNode head) {
        if (head == null) return null;
        
        ListNode current = head.next;
        head.next = null;
        
        while (current != null) {
            ListNode next = current.next;
            current.next = head;
            head = current;
            current = next;
        }
        
        return head;
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
	 * String to Integer (atoi)
	 * 
	 * Implement atoi to convert a string to an integer.
	 * 
	 * Hint: Carefully consider all possible input cases. If you want a
	 * challenge, please do not see below and ask yourself what are the possible
	 * input cases.
	 * 
	 * Notes: It is intended for this problem to be specified vaguely (ie, no
	 * given input specs). You are responsible to gather all the input
	 * requirements up front.
	 */
    public int atoi(String str) {
        int len = str.length();
        int iter = 0;
        boolean negative = false;
        int result = 0;
        
        // delete white space
        while (iter < len && str.charAt(iter) == ' ') {
            iter++;
        }
        if (iter == len) return 0;
        
        // check sign
        if (str.charAt(iter) == '-') {
            negative = true;
            iter++;
        } else if (str.charAt(iter) == '+') {
            iter++;
        }
        
        // find integer
        while (iter < len) {
            int digit = str.charAt(iter) - '0';
            if (digit < 0 || digit > 9) {
                break;
            } else {
                if (negative) {
                    int bound = (Integer.MIN_VALUE + digit) / 10;
                    if (result < bound) {
                        return Integer.MIN_VALUE;
                    } else {
                        result = result * 10 - digit;
                    }
                } else {
                    int bound = (Integer.MAX_VALUE - digit) / 10;
                    if (result > bound) {
                        return Integer.MAX_VALUE;
                    } else {
                        result = result * 10 + digit;
                    }
                }
            }
            iter++;
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
	 * Longest Palindromic Substring
	 * 
	 * Given a string S, find the longest palindromic substring in S. You may
	 * assume that the maximum length of S is 1000, and there exists one unique
	 * longest palindromic substring.
	 */
	public String longestPalindrome(String s) {
		int len = s.length();
        if (len < 2) {
            return s;
        }
        
        int left = 0;
        int right = 0;
        int maxLen = 1;
        for (int i = 1; i < len - 1; i++) {
        	// no longer substring can be found
        	if (2 * i + 1 <= maxLen || (len - i) * 2 <= maxLen) break;
        	
            int l, r;
            // even number of characters
            for (l = i - 1, r = i; l >= 0 && r < len; l--, r++) {
                if (s.charAt(l) != s.charAt(r)) {
                    break;
                }
            }
            l++;
            r--;
            if (r - l + 1 > maxLen) {
                left = l;
                right = r;
                maxLen = right - left + 1;
            }
            
            // odd number of characters
            for (l = i - 1, r = i + 1; l >= 0 && r < len; l--, r++) {
                if (s.charAt(l) != s.charAt(r)) {
                    break;
                }
            }
            l++;
            r--;
            if (r - l + 1 > maxLen) {
                left = l;
                right = r;
                maxLen = right - left + 1;
            }
        }
        
        if (maxLen == 1 && s.charAt(len - 1) == s.charAt(len - 2)) {
            return s.substring(len - 2);
        }
        
        return s.substring(left, right + 1);
    }
	
	/**
	 * Edit Distance
	 * 
	 * Given two words word1 and word2, find the minimum number of steps
	 * required to convert word1 to word2. (each operation is counted as 1
	 * step.)
	 * 
	 * You have the following 3 operations permitted on a word:
	 * a) Insert a character b) Delete a character c) Replace a character
	 */
	public int minDistance(String word1, String word2) {
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
     * Decode Ways
     */
    public int numDecodings(String s) {
        int len = s.length();
        if (len == 0) return 0;
        
        int[] dp = new int[len + 1];
        dp[0] = 1;
        dp[1] = (s.charAt(0) == '0') ? 0 : 1;
        
        for (int i = 2; i <= len; i++) {
            int n = s.charAt(i - 1) - '0';
            if (n > 0) {
                dp[i] = dp[i - 1];
            }
            
            n += (s.charAt(i - 2) - '0') * 10;
            if (n <= 26 && n >= 10) {
                dp[i] += dp[i - 2];
            }
        }
        
        return dp[len];
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
     * Copy List with Random Pointer
     */
    public RandomListNode copyRandomList(RandomListNode head) {
        if (head == null) {
            return null;
        }
        
        // copy the main list to double the length
        RandomListNode current = head;
        while (current != null) {
            RandomListNode node = new RandomListNode(current.label);
            node.next = current.next;
            current.next = node;
            node.random = current.random;
            current = node.next;
        }
        
        // update the random
        current = head;
        while (current != null) {
            current = current.next;
            current.random = (current.random == null) ? null : current.random.next;
            current = current.next;
        }

        // separate the two lists
        RandomListNode newHead = head.next;
        current = newHead;
        while (current.next != null) {
            head.next = current.next;
            head = head.next;
            current.next = head.next;
            current = current.next;
        }
        head.next = null;
        
        return newHead;
    }
    
    /**
     * Merge k Sorted Lists
     */
    public ListNode mergeKLists(ArrayList<ListNode> lists) {
        if (lists == null || lists.size() == 0) {
            return null;
        }

        // construct dummy node
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;
        
        // construct MinHeap
        PriorityQueue<ListNode> pq = new PriorityQueue<ListNode>(lists.size(), 
                new Comparator<ListNode>() {
                    @Override
                    public int compare(ListNode node1, ListNode node2) {
                        if (node1.val < node2.val) {
                            return -1;
                        } else if (node1.val > node2.val) {
                            return 1;
                        }
                        
                        return 0;
                    }
                });
        
        // add all head node to heap
        for (ListNode node : lists) {
            if (node != null) { // this is ridiculous
                pq.offer(node);
            }
        }
        
        while (!pq.isEmpty()) {
            current.next = pq.poll();
            current = current.next;
            if (current.next != null) {
                pq.offer(current.next);
            }
        }
        
        return dummy.next;
    }
    
    /**
     * Sort List
     * 
     * Sort a linked list in O(n log n) time using constant space complexity.
     */
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        
        // divide
        ListNode tail = findMiddle(head);
        ListNode head2 = tail.next;
        tail.next = null;
        
        // conquer
        head = sortList(head);
        head2 = sortList(head2);
        return merge(head, head2);
    }
    
    private ListNode findMiddle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head.next;
        
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        
        return slow;
    }
    
    private ListNode merge(ListNode head1, ListNode head2) {
        ListNode dummy = new ListNode(0);
        ListNode prev = dummy;
        
        while (head1 != null && head2 != null) {
            
            if (head1.val < head2.val) {
                prev.next = head1;
                prev = head1;
                head1 = head1.next;
            } else {
                prev.next = head2;
                prev = head2;
                head2 = head2.next;
            }
        }
        
        prev.next = (head1 == null) ? head2 : head1;
        
        return dummy.next;
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
	 * Clone Graph
	 */
	public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
		HashMap<UndirectedGraphNode, UndirectedGraphNode> map = new HashMap<UndirectedGraphNode, UndirectedGraphNode>();
		return cloneNode(node, map);
	}

	private UndirectedGraphNode cloneNode(UndirectedGraphNode node,
			HashMap<UndirectedGraphNode, UndirectedGraphNode> map) {
		if (node == null)
			return null;
		if (map.containsKey(node)) { // have copied before
			return map.get(node);
		} else { // hasn't been copied
			UndirectedGraphNode copy = new UndirectedGraphNode(node.label);
			map.put(node, copy); // put the new copy into map
			// add copies of children
			for (UndirectedGraphNode n : node.neighbors) {
				copy.neighbors.add(cloneNode(n, map));
			}
			return copy;
		}
	}
    
    /**
     * Surrounded Regions
     */
    public void solve(char[][] board) {
		if (board.length == 0) {
			return;
		}
		boolean isAlive[][] = new boolean[board.length][board[0].length];

		for (int i = 1; i < board[0].length - 1; i++) {
			dfs(0, i, isAlive, board);
			dfs(board.length - 1, i, isAlive, board);
		}

		for (int i = 1; i < board.length - 1; i++) {
			dfs(i, 0, isAlive, board);
			dfs(i, board[0].length - 1, isAlive, board);
		}

		for (int i = 1; i < board.length - 1; i++) {
			for (int j = 1; j < board[0].length - 1; j++) {
				if (!isAlive[i][j]) {
					board[i][j] = 'X';
				}
			}
		}
	}

	public void dfs(int row, int col, boolean[][] isAlive, char[][] board) {
		if (board[row][col] == 'X' || isAlive[row][col]) {
			return;
		} else if (board[row][col] == 'O') {
			isAlive[row][col] = true;
		}

		if (col - 1 > 0) {
			dfs(row, col - 1, isAlive, board);
		}
		if (col + 1 < isAlive[0].length - 1) {
			dfs(row, col + 1, isAlive, board);
		}
		if (row - 1 > 0) {
			dfs(row - 1, col, isAlive, board);
		}
		if (row + 1 < isAlive.length - 1) {
			dfs(row + 1, col, isAlive, board);
		}
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
	 * Regular Expression Matching
	 */
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
				while (i <= s.length() && (p.charAt(0) == '.' || s.charAt(i - 1) == p.charAt(0))) {
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
		System.out.println(numDecodings("650"));
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
