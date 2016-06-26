package chapter_4_TreesAndGraphs;

import CareerCupLibrary.TreeNode;

public class TreesAndGraphs {
	// ************************** HELPER FUNCTIONS **************************
	private static TreeNode buildBST() {
		TreeNode root = new TreeNode(20);
		root.setLeftChild(new TreeNode(10));
		root.setRightChild(new TreeNode(30));
		root.left.setLeftChild(new TreeNode(5));
		root.left.setRightChild(new TreeNode(15));
		root.left.left.setLeftChild(new TreeNode(3));
		root.left.left.setRightChild(new TreeNode(7));
		root.left.right.setRightChild(new TreeNode(17));
		return root;
	}
	
	private static TreeNode buildBinaryTree() {
		TreeNode root = new TreeNode(20);
		root.setLeftChild(new TreeNode(10));
		root.setRightChild(new TreeNode(30));
		root.left.setLeftChild(new TreeNode(5));
		root.left.setRightChild(new TreeNode(15));
		root.right.setRightChild(new TreeNode(25));
		root.left.left.setLeftChild(new TreeNode(3));
		root.left.left.setRightChild(new TreeNode(7));
		root.left.right.setRightChild(new TreeNode(17));
		return root;
	}
	
	private static TreeNode internalNode() {
		return buildBST().left.right.right;
	}

	// ************************** PROBLEMS SOLUTIONS **************************
	/**
	 * 4.5 Check if a tree is binary search tree.
	 * 
	 * @param n
	 * @param min
	 * @param max
	 * @return
	 */
	private static boolean isBST(TreeNode n, int min, int max) {
		if (n == null) {
			return true;
		}
		if (n.data < min || n.data > max) {
			return false;
		}
		return isBST(n.left, min, n.data) && isBST(n.right, n.data, max);
	}
	
	public static boolean isBinarySearchTree(TreeNode root) {
		return isBST(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
	}

	public static void testIsBinarySearchTree() {
		System.out.println("4.5 testing isBinarySearchTree()...");
		System.out.println(isBinarySearchTree(buildBST()));
		System.out.println(isBinarySearchTree(buildBinaryTree()));
		System.out.println();
	}
	
	
	/**
	 * 4.6 Find the 'next' node (in-order successor) of a given node. This
	 * problem indicates pseudo code is helpful.
	 * 
	 * @param n
	 * @return
	 */
	public static TreeNode inorderSucc(TreeNode n) {
		if (n == null) {
			return null;
		}
		
		if (n.right != null) {
			TreeNode parent = n.right;
			while (parent.left != null) {
				parent = parent.left;
			}
			return parent;
		}
		
		while (n.parent != null && n.parent.right == n) {
			n = n.parent;
		}
		
		return n.parent;
	}
	
	public static void testInorderSucc() {
		System.out.println("4.6 testing inorderSucc()...");
		TreeNode n = internalNode();
		System.out.println(n.data + " -> " + inorderSucc(n).data);
		System.out.println();
	}
	
	
	/**
	 * 4.7 Find the first common ancestor of two nodes in a binary tree. Avoid
	 * storing additional nodes in a data structure. NOTE: This is not necessary
	 * a binary search tree.
	 * 
	 * @param n1
	 * @param n2
	 * @return
	 */
	private static boolean covers(TreeNode parent, TreeNode child) {
		if (parent == null) {
			return false;
		} else if (parent == child) {
			return true;
		} else {
			return covers(parent.left, child) || covers(parent.right, child);
		}
	}

	public static TreeNode firstCommonAncestor1(TreeNode n1, TreeNode n2) {
		if (n1 == null || n2 == null) {
			return null;
		}
		
		while (n1.parent != null) {
			if (covers(n1.parent, n2)) {
				return n1.parent;
			}
			n1 = n1.parent;
		}
		
		return null;
	}
	
	public static TreeNode firstCommonAncestor(TreeNode n1, TreeNode n2) {
		if (n1 == null || n2 == null) {
			return null;
		}
		
		TreeNode node;
		
		node = n1;
		int depth1 = 0;
		while (node.parent != null) {
			node = node.parent;
			depth1++;
		}

		node = n2;
		int depth2 = 0;
		while (node.parent != null) {
			node = node.parent;
			depth2++;
		}
		
		while (depth1 > depth2) {
			n1 = n1.parent;
			depth1--;
		}
		while (depth1 < depth2) {
			n2 = n2.parent;
			depth2--;
		}
		
		while (n1.parent != null && n2.parent != null) {
			if (n1 == n2) {
				return n1;
			}
			n1 = n1.parent;
			n2 = n2.parent;
		}
		return null;
	}
	
	public static void testFirstCommonAncestor() {
		System.out.println("4.7 testing firstCommonAncestor()...");
		TreeNode root = buildBST();
		TreeNode n1 = root.left.left.right;
		TreeNode n2 = root.left.right.right;
		System.out.println(n1.data + " ->");
		System.out.println("      " + firstCommonAncestor(n1, n2).data);
		System.out.println(n2.data + " ->");
		System.out.println();
	}
	
	
	/**
	 * 4.8 You have two very large binary trees: T1, with millions of nodes, and
	 * T2, with hundreds of nodes. Create an algorithm to decide if T2 is a
	 * subtree of T1.
	 * 
	 * @param r1
	 * @param r2
	 * @return
	 */
	public static boolean containsTree(TreeNode r1, TreeNode r2) {
		if (matchTrees(r1, r2)) {
			return true;
		} else if (r1 == null) {
			return false;
		} else {
			return containsTree(r1.left, r2) || containsTree(r1.right, r2);
		}
	}
	
	public static boolean matchTrees(TreeNode r1, TreeNode r2) {
		if (r1 == null && r2 == null) {
			return true;
		} else if (r1 == null || r2 == null) {
			return false;
		} else if (r1.data != r2.data) {
			return false;
		} else {
			return matchTrees(r1.left, r2.left) && matchTrees(r1.right, r2.right);
		}
	}
	
	public static void testContainsTree() {
		System.out.println("4.8 testing containsTree()...");
		TreeNode r1 = buildBST();
		TreeNode r2 = buildBST();
		System.out.println("r1 contains r2: " + containsTree(r1, r2));
		
		r2.right = null;
		System.out.println("After changing r2...\nr1 contains r2: " + containsTree(r1, r2));
	}
	
	/**
	 * 4.9 Given a binary tree in which each node contains a value. Design an
	 * algorithm to print all paths which sum to a given value. Note that a path
	 * can start or end anywhere in the tree.
	 * 
	 * @param r
	 * @param sum
	 */
	public static void findSum(TreeNode r, int sum) {
		int[] path = new int[findDepth(r)];
		int level = 0;
		findSumUtil(r, sum, path, level);
	}
	
	/** recursively find the path end at n */
	public static void findSumUtil(TreeNode n, int sum, int[] path, int level) {
		if (n == null) {
			return;
		} else {
			path[level] = n.data;
			
			int count = 0;
			for (int i = level; i >= 0; i--) {
				count += path[i];
				if (count == sum) {
					printPath(sum, path, i, level);
				}
			}
			findSumUtil(n.left, sum, path, level + 1);
			findSumUtil(n.right, sum, path, level + 1);
		}
	}
	
	/** @return depth of a tree starts from r */
	public static int findDepth(TreeNode r) {
		if (r == null) {
			return 0;
		} else {
			return 1 + Math.max(findDepth(r.left), findDepth(r.right));
		}
	}
	
	/** print the path */
	public static void printPath(int sum, int[] path, int start, int end) {
		System.out.print("path(" + sum + "): ");
		for (int i = start; i < end; i++) {
			System.out.print(path[i] + " -> ");
		}
		System.out.println(path[end]);
	}
	
	public static void testFindSum() {
		System.out.println("4.9 testing findSum()...");
		
		TreeNode r = buildBinaryTree();
		int sum = 62;
		
		findSum(r, sum);
	}
	
	/**
	 * test all functions
	 */
	public static void testAll() {
		testIsBinarySearchTree();
		testInorderSucc();
		testFirstCommonAncestor();
		testContainsTree();
		testFindSum();
	}
	
	public static void main(String[] args) {
		testAll();
	}

}
