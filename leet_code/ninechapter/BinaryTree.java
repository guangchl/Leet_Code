package ninechapter;

import java.util.ArrayList;

public class BinaryTree {
	/** Definition for binary tree */
	public class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;

		TreeNode(int x) {
			val = x;
		}
	}
	
	/** Definition for singly-linked list. */
    public class ListNode {
        int val;
        ListNode next;
    
        public ListNode(int x) {
            val = x;
            next = null;
        }
    }

	/**
	 * 1. Binary Tree Preorder Traversal
	 */
	public ArrayList<Integer> preorderTraversalRec(TreeNode root) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		preorderTraversalHelper(root, result);
		return result;
	}
	
	private void preorderTraversalHelper(TreeNode root, ArrayList<Integer> result) {
		if (root == null) {
			return;
		}
		
		result.add(root.val);
		preorderTraversalHelper(root.left, result);
		preorderTraversalHelper(root.right, result);
	}

	public ArrayList<Integer> preorderTraversal(TreeNode root) {
		return null;
	}
	
	/**
	 * Convert Sorted Array to Binary Search Tree
	 * 
	 * Given an array where elements are sorted in ascending order, convert it
	 * to a height balanced BST.
	 */
	private TreeNode buildTree(int[] num, int start, int end) {
		if (start > end) {
			return null;
		}
		
		TreeNode node = new TreeNode(num[(start + end) / 2]);
		
		node.left = buildTree(num, start, (start + end) / 2 - 1);
		node.right = buildTree(num, (start + end) / 2 + 1, end);
		
		return node;
	}

	public TreeNode sortedArrayToBST(int[] num) {
		if (num == null) {
			return null;
		}

		return buildTree(num, 0, num.length - 1);
	}

	/**
	 * Convert Sorted List to Binary Search Tree
	 * 
	 * Given a singly linked list where elements are sorted in ascending order,
	 * convert it to a height balanced BST.
	 */
    private ListNode currentHead;
    public TreeNode sortedListToBST(ListNode head) {
        int size;

        currentHead = head;
        size = getListLength(head);

        return sortedListToBSTHelper(size);
    }

    private int getListLength(ListNode head) {
        int size = 0;

        while (head != null) {
            size++;
            head = head.next;
        }

        return size;
    }

    private TreeNode sortedListToBSTHelper(int size) {
        if (size <= 0) {
            return null;
        }

        TreeNode left = sortedListToBSTHelper(size / 2);
        TreeNode root = new TreeNode(currentHead.val);
        currentHead = currentHead.next;
        TreeNode right = sortedListToBSTHelper((size - 1) / 2);

        root.left = left;
        root.right = right;

        return root;
    }
}
