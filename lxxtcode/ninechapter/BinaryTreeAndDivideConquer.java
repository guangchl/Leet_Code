package ninechapter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Stack;

import ninechapter.BinaryTreeAndDivideConquer.TreeNode;

public class BinaryTreeAndDivideConquer {

    // **************************** Data Structure ****************************

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

    // ******************************* TRAVERSAL *******************************

    /**
     * Binary Tree Preorder Traversal
     *
     * Given a binary tree, return the preorder traversal of its nodes' values.
     *
     * For example: Given binary tree {1,#,2,3}, return [1, 2, 3]
     *
     * Note: Recursive solution is trivial, could you do it iteratively?
     *
     * @param root:
     *            The root of binary tree.
     * @return: Preorder in ArrayList which contains node values.
     */
    public ArrayList<Integer> preorderTraversalRec(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        preorderTraverse(root, result);
        return result;
    }

    private void preorderTraverse(TreeNode root, List<Integer> result) {
        if (root != null) {
            result.add(root.val);
            preorderTraverse(root.left, result);
            preorderTraverse(root.right, result);
        }
    }

    /**
     * Binary Tree Preorder Traversal.
     *
     * Iterative solution.
     *
     * @param root:
     *            The root of binary tree.
     * @return: Preorder in ArrayList which contains node values.
     */
    public ArrayList<Integer> preorderTraversalIter(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);

        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            if (node != null) {
                result.add(node.val);
                stack.push(node.right);
                stack.push(node.left);
            }
        }

        return result;
    }

    /**
     * Binary Tree Level Order Traversal.
     *
     * Given a binary tree, return the level order traversal of its nodes'
     * values. (ie, from left to right, level by level).
     *
     * @param root:
     *            The root of binary tree.
     * @return: Level order a list of lists of integer
     */
    @tags.Queue
    @tags.BinaryTree
    @tags.BFS
    @tags.BinaryTreeTraversal
    @tags.Company.Facebook
    @tags.Company.LinkedIn
    @tags.Company.Uber
    public ArrayList<ArrayList<Integer>> levelOrder(TreeNode root) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (root == null) {
            return result;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        queue.offer(null);
        ArrayList<Integer> level = new ArrayList<>();

        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node != null) {
                level.add(node.val);
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            } else {
                result.add(level);
                if (queue.isEmpty()) {
                    break;
                } else {
                    queue.offer(null);
                    level = new ArrayList<>();
                }
            }
        }

        return result;
    }

    /**
     * Binary Tree Level Order Traversal II
     *
     * Given a binary tree, return the bottom-up level order traversal of its
     * nodes' values. (ie, from left to right, level by level from leaf to
     * root).
     */
    @tags.Queue
    @tags.BinaryTree
    @tags.BinaryTreeTraversal
    @tags.BFS
    public ArrayList<ArrayList<Integer>> levelOrderBottom(TreeNode root) {
        ArrayList<ArrayList<Integer>> result = levelOrder(root);
        Collections.reverse(result);
        return result;
    }

    // ******************************* PROBLEMS *******************************

    /**
     * Maximum Depth of Binary Tree.
     *
     * Given a binary tree, find its maximum depth. The maximum depth is the
     * number of nodes along the longest path from the root node down to the
     * farthest leaf node.
     *
     * @param root:
     *            The root of binary tree.
     * @return: An integer.
     */
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }

        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    /**
     * Balanced Binary Tree.
     *
     * Given a binary tree, determine if it is height-balanced.
     *
     * For this problem, a height-balanced binary tree is defined as a binary
     * tree in which the depth of the two subtrees of every node never differ by
     * more than 1.
     *
     * @param root:
     *            The root of binary tree.
     * @return: True if this Binary tree is Balanced, or false.
     */
    public boolean isBalanced(TreeNode root) {
        return height(root) != -1;
    }

    private int height(TreeNode node) {
        if (node == null) {
            return 0;
        }

        int leftH = height(node.left);
        int rightH = height(node.right);

        if (leftH == -1 || rightH == -1 || Math.abs(leftH - rightH) > 1) {
            return -1;
        }

        return Math.max(leftH, rightH) + 1;
    }

    /**
     * Binary Tree Maximum Path Sum.
     *
     * Given a binary tree, find the maximum path sum. The path may start and
     * end at any node in the tree.
     *
     * @param root:
     *            The root of binary tree.
     * @return: An integer.
     */
    @tags.DivideAndConquer
    @tags.DanymicProgramming
    @tags.Recursion
    public int maxPathSum(TreeNode root) {
        return longestPathToNode(root).max;
    }

    public ResultTypeMPS longestPathToNode(TreeNode node) {
        if (node == null) {
            return new ResultTypeMPS(0, Integer.MIN_VALUE);
        }

        ResultTypeMPS left = longestPathToNode(node.left);
        ResultTypeMPS right = longestPathToNode(node.right);
        int leftPath = Math.max(left.path, 0);
        int rightPath = Math.max(right.path, 0);

        // get the path
        int path = node.val + Math.max(leftPath, rightPath);

        // get the max
        int max = Math.max(left.max, right.max);
        int fullPath = node.val + leftPath + rightPath;
        max = Math.max(max, fullPath);

        return new ResultTypeMPS(path, max);
    }

    private class ResultTypeMPS {
        int path;
        int max;

        ResultTypeMPS(int path, int max) {
            this.path = path;
            this.max = max;
        }
    }

    /**
     * Binary Tree Maximum Path Sum II.
     *
     * @param root
     *            the root of binary tree.
     * @return an integer
     */
    @tags.BinaryTree
    public int maxPathSum2(TreeNode root) {
        if (root != null) {
            int leftMax = maxPathSum2(root.left);
            int rightMax = maxPathSum2(root.right);
            int childMax = Math.max(leftMax, rightMax);

            return root.val + Math.max(childMax, 0);
        }
        return 0;
    }

    /**
     * Inorder Successor in Binary Search Tree.
     *
     * Given a binary search tree (See Definition) and a node in it, find the
     * in-order successor of that node in the BST.
     *
     * If the given node has no in-order successor in the tree, return null.
     *
     * Notice: It's guaranteed p is one node in the given tree. (You can
     * directly compare the memory address to find p)
     *
     * @param root
     * @param p
     * @return
     */
    @tags.BinaryTree
    @tags.BinarySearchTree
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode current = root;
        boolean foundP = false;

        while (current != null || !stack.isEmpty()) {
            if (current == null) {
                do {
                    current = stack.pop();
                } while (current == null);

                if (!foundP) {
                    foundP = (current.val == p.val);
                } else {
                    return current;
                }
                current = current.right;
            } else {
                stack.push(current);
                current = current.left;
            }
        }

        return null;
    }

    /**
     * Validate Binary Search Tree.
     *
     * Given a binary tree, determine if it is a valid binary search tree (BST).
     *
     * Assume a BST is defined as follows:
     *
     * The left subtree of a node contains only nodes with keys less than the
     * node's key. The right subtree of a node contains only nodes with keys
     * greater than the node's key. Both the left and right subtrees must also
     * be binary search trees. A single node tree is a BST
     *
     * @param root:
     *            The root of binary tree.
     * @return: True if the binary tree is BST, or false
     */
    @tags.BinaryTree
    @tags.BinarySearchTree
    @tags.DivideAndConquer
    @tags.Recursion
    public boolean isValidBST(TreeNode root) {
        return isValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    boolean isValidBST(TreeNode node, long min, long max) {
        if (node == null) {
            return true;
        }

        if (min < node.val && node.val < max) {
            boolean left = isValidBST(node.left, min, node.val);
            boolean right = isValidBST(node.right, node.val, max);
            return left && right;
        }
        return false;
    }

    /**
     * Lowest Common Ancestor.
     *
     * Given the root and two nodes in a Binary Tree. Find the lowest common
     * ancestor(LCA) of the two nodes. The lowest common ancestor is the node
     * with largest depth which is the ancestor of both nodes.
     *
     * @param root:
     *            The root of the binary search tree.
     * @param A
     *            and B: two nodes in a Binary.
     * @return: Return the least common ancestor(LCA) of the two nodes.
     */
    @tags.BinaryTree
    @tags.Company.Facebook
    @tags.Company.LinkedIn
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode A, TreeNode B) {
        if (root == null || root == A || root == B) {
            return root;
        }

        // divide
        TreeNode left = lowestCommonAncestor(root.left, A, B);
        TreeNode right = lowestCommonAncestor(root.right, A, B);

        // conquer: merge A and B at the LCA
        if (left != null && right != null) {
            return root;
        }

        // send A or B upward
        if (left != null) {
            return left;
        }
        if (right != null) {
            return right;
        }

        // nothing found yet
        return null;
    }

    // ------------------------ OLD ------------------------------------

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

    public void test () {
        TreeNode root = new TreeNode(1);
        TreeNode left = new TreeNode(1);
        TreeNode right = new TreeNode(3);
        root.left = left;
        root.right = right;

        levelOrder(root);
    }

    public static void main(String[] args) {
        BinaryTreeAndDivideConquer btdc = new BinaryTreeAndDivideConquer();
        btdc.test();
    }

}

// ---------------------------- Other ----------------------------

/**
 * Binary Search Tree Iterator
 *
 * Design an iterator over a binary search tree with the following rules: 1.
 * Elements are visited in ascending order (i.e. an in-order traversal). 2.
 * next() and hasNext() queries run in O(1) time in average.
 */
@tags.BinaryTree
@tags.NonRecursion
@tags.BinarySearchTree
@tags.Company.Facebook
@tags.Company.Google
@tags.Company.LinkedIn
class BSTIterator {
    TreeNode current;
    Stack<TreeNode> stack = new Stack<TreeNode>();

    // @param root: The root of binary tree.
    public BSTIterator(TreeNode root) {
        this.current = root;
    }

    // @return: True if there has next node, or false
    public boolean hasNext() {
        if (current != null || !stack.isEmpty()) {
            return true;
        }
        return false;
    }

    // @return: return next node
    public TreeNode next() {
        while (current != null) {
            stack.push(current);
            current = current.left;
        }

        TreeNode min = stack.pop();
        if (min.right != null) {
            current = min.right;
        }

        return min;
    }
}
