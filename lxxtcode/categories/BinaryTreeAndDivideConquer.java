package categories;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Queue;
import java.util.Stack;

import org.junit.Test;

public class BinaryTreeAndDivideConquer {

    // ---------------------------------------------------------------------- //
    // ------------------------------- MODELS ------------------------------- //
    // ---------------------------------------------------------------------- //

    /** Definition for binary tree */
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    /** Definition for binary tree with parent pointer */
    class ParentTreeNode {
        public ParentTreeNode parent, left, right;
    }

    /** Definition for singly-linked list. */
    class ListNode {
        int val;
        ListNode next;

        public ListNode(int x) {
            val = x;
            next = null;
        }
    }

    /** Definition for binary tree with next pointer. */
    public class TreeLinkNode {
        int val;
        TreeLinkNode left, right, next;

        TreeLinkNode(int x) {
            val = x;
        }
    }

    // ---------------------------------------------------------------------- //
    // ----------------------------- TRAVERSAL ------------------------------ //
    // ---------------------------------------------------------------------- //

    /**
     * Binary Tree Preorder Traversal - recursive solution.
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
    @tags.Recursion
    @tags.Tree
    @tags.BinaryTree
    @tags.BinaryTreeTraversal
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
     * Binary Tree Preorder Traversal - iterative solution.
     *
     * @param root:
     *            The root of binary tree.
     * @return: Preorder in ArrayList which contains node values.
     */
    @tags.NonRecursion
    @tags.Stack
    @tags.Tree
    @tags.BinaryTree
    @tags.BinaryTreeTraversal
    @tags.Status.Easy
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
     * Binary Tree Postorder Traversal.
     *
     * Example: Given binary tree {1,#,2,3}, return [3,2,1].
     *
     * Challenge: Can you do it without recursion?
     *
     * @param root:
     *            The root of binary tree.
     * @return: Postorder in ArrayList which contains node values.
     */
    @tags.Recursion
    @tags.Stack
    @tags.Tree
    @tags.BinaryTree
    @tags.BinaryTreeTraversal
    @tags.Status.Hard
    public ArrayList<Integer> postorderTraversal(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode current = root;
        TreeNode prev = null;

        while (current != null || !stack.isEmpty()) {
            while (current != null) {
                stack.push(current);
                current = current.left;
            }

            if (stack.peek().right == prev || stack.peek().right == null) {
                prev = stack.pop();
                result.add(prev.val);
            } else {
                current = stack.peek().right;
            }
        }

        return result;
    }

    /**
     * Binary Tree Inorder Traversal.
     *
     * Given a binary tree, return the inorder traversal of its nodes' values.
     *
     * Example: Given binary tree {1,#,2,3}, return [1,3,2].
     *
     * Challenge: Can you do it without recursion?
     *
     * @param root:
     *            The root of binary tree.
     * @return: Inorder in ArrayList which contains node values.
     */
    @tags.Recursion
    @tags.Tree
    @tags.BinaryTree
    @tags.BinaryTreeTraversal
    @tags.Company.Microsoft
    @tags.Status.OK
    public ArrayList<Integer> inorderTraversal(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode current = root;

        while (current != null || !stack.isEmpty()) {
            while  (current != null) {
                stack.push(current);
                current = current.left;
            }

            current = stack.pop();
            result.add(current.val);
            current = current.right;
        }

        return result;
    }

    /**
     * Binary Tree Level Order Traversal.
     *
     * Given a binary tree, return the level order traversal of its nodes'
     * values. (ie, from left to right, level by level).
     *
     * Example: Given binary tree {3,9,20,#,#,15,7}, return its level order
     * traversal as: [ [3], [9,20], [15,7] ].
     *
     * Challenge: Challenge 1: Using only 1 queue to implement it. Challenge 2:
     * Use DFS algorithm to do it.
     *
     * @param root:
     *            The root of binary tree.
     * @return: Level order a list of lists of integer
     */
    @tags.Queue
    @tags.Tree
    @tags.BinaryTree
    @tags.BFS
    @tags.BinaryTreeTraversal
    @tags.Company.Amazon
    @tags.Company.Apple
    @tags.Company.Bloomberg
    @tags.Company.Facebook
    @tags.Company.LinkedIn
    @tags.Company.Microsoft
    @tags.Company.Uber
    @tags.Status.OK
    public ArrayList<ArrayList<Integer>> levelOrder(TreeNode root) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (true) {
            Queue<TreeNode> newQueue = new LinkedList<>();
            ArrayList<Integer> level = new ArrayList<>();

            for (TreeNode node : queue) {
                if (node != null) {
                    level.add(node.val);
                    newQueue.offer(node.left);
                    newQueue.offer(node.right);
                }
            }

            if (level.isEmpty()) {
                break;
            }

            result.add(level);
            queue = newQueue;
        }

        return result;
    }

    /**
     * Binary Tree Level Order Traversal II.
     *
     * Given a binary tree, return the bottom-up level order traversal of its
     * nodes' values. (ie, from left to right, level by level from leaf to
     * root).
     *
     * Example: Given binary tree {3,9,20,#,#,15,7}, return its level order
     * traversal as: [ [15,7], [9,20], [3] ].
     *
     * @param root: The root of binary tree.
     * @return: buttom-up level order a list of lists of integer
     */
    @tags.Queue
    @tags.Tree
    @tags.BinaryTree
    @tags.BinaryTreeTraversal
    @tags.BFS
    public ArrayList<ArrayList<Integer>> levelOrderBottom(TreeNode root) {
        ArrayList<ArrayList<Integer>> result = levelOrder(root);
        Collections.reverse(result);
        return result;
    }

    /**
     * Binary Tree Zigzag Level Order Traversal.
     *
     * Given a binary tree, return the zigzag level order traversal of its
     * nodes' values. (ie, from left to right, then right to left for the next
     * level and alternate between).
     *
     * Example: Given binary tree {1,2,3,4,#,#,5,#,#,6,7,#,#,#,8}, return its
     * zigzag level order traversal as: [[1],[3,2],[4,5],[7,6],[8]].
     *
     * @param root:
     *            The root of binary tree.
     * @return: A list of lists of integer include the zigzag level order
     *          traversal of its nodes' values
     */
    @tags.Stack
    @tags.Queue
    @tags.BFS
    @tags.Tree
    @tags.BinaryTree
    @tags.BinaryTreeTraversal
    @tags.Company.Bloomberg
    @tags.Company.LinkedIn
    @tags.Company.Microsoft
    public ArrayList<ArrayList<Integer>> zigzagLevelOrder(TreeNode root) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        boolean forward = true;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);

        while (true) {
            ArrayList<Integer> level = new ArrayList<>();
            Stack<TreeNode> newStack = new Stack<>();

            while (!stack.isEmpty()) {
                TreeNode node = stack.pop();
                if (node != null) {
                    level.add(node.val);
                    if (forward) {
                        newStack.push(node.left);
                        newStack.push(node.right);
                    } else {
                        newStack.push(node.right);
                        newStack.push(node.left);
                    }
                }
            }

            if (level.isEmpty()) {
                break;
            }
            result.add(level);
            forward = !forward;
            stack = newStack;
        }

        return result;
    }

    /**
     * Binary Tree Vertical Order Traversal.
     *
     * Given a binary tree, return the vertical order traversal of its nodes'
     * values. (ie, from top to bottom, column by column).
     *
     * If two nodes are in the same row and column, the order should be from
     * left to right.
     *
     * @param root
     * @return
     */
    @tags.HashTable
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Company.Snapchat
    public List<List<Integer>> verticalOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        Map<Integer, List<Integer>> map = new HashMap<>();
        int[] minMax = new int[2]; // left most and right most

        class NodePos {
            TreeNode node;
            int pos;

            public NodePos(TreeNode node, int pos) {
                this.node = node;
                this.pos = pos;
            }
        }

        Queue<NodePos> queue = new LinkedList<>();
        queue.add(new NodePos(root, 0));

        // bfs to traverse every node level by level and left to right
        while (!queue.isEmpty()) {
            Queue<NodePos> next = new LinkedList<>();

            for (NodePos node : queue) {
                if (node.node != null) {
                    if (!map.containsKey(node.pos)) {
                        map.put(node.pos, new ArrayList<Integer>());
                        minMax[0] = Math.min(minMax[0], node.pos);
                        minMax[1] = Math.max(minMax[1], node.pos);
                    }

                    map.get(node.pos).add(node.node.val);

                    next.offer(new NodePos(node.node.left, node.pos - 1));
                    next.offer(new NodePos(node.node.right, node.pos + 1));
                }
            }

            queue = next;
        }

        // add list from map to result, from left to right
        for (int i = minMax[0]; i <= minMax[1]; i++) {
            if (map.containsKey(i)) {
                result.add(map.get(i));
            }
        }

        return result;
    }

    // ---------------------------------------------------------------------- //
    // ------------------------------ PROBLEMS ------------------------------ //
    // ---------------------------------------------------------------------- //

    /**
     * Sum of Left Leaves.
     *
     *Find the sum of all left leaves in a given binary tree.
     *
     * @param root
     * @return
     */
    @tags.Tree
    @tags.Company.Facebook
    public int sumOfLeftLeaves(TreeNode root) {
        if (root == null) {
            return 0;
        }

        return sumOfLeftLeaves(root, false);
    }

    private int sumOfLeftLeaves(TreeNode node, boolean isLeft) {
        if (node.left == null && node.right == null) {
            return isLeft ? node.val : 0;
        }

        int sum = 0;

        if (node.left != null) {
            sum += sumOfLeftLeaves(node.left, true);
        }
        if (node.right != null) {
            sum += sumOfLeftLeaves(node.right, false);
        }

        return sum;
    }

    /**
     * Clone Binary Tree.
     *
     * For the given binary tree, return a deep copy of it.
     *
     * @param root:
     *            The root of binary tree
     * @return root of new tree
     */
    @tags.BinaryTree
    @tags.Recursion
    @tags.Status.Easy
    public TreeNode cloneTree(TreeNode root) {
        if (root == null) {
            return null;
        }

        TreeNode clone = new TreeNode(root.val);
        clone.left = cloneTree(root.left);
        clone.right = cloneTree(root.right);
        return clone;
    }

    /**
     * Binary Tree Paths.
     *
     * Given a binary tree, return all root-to-leaf paths.
     *
     * Example: Given the following binary tree: [1, 2, 3, #, 5]. All
     * root-to-leaf paths are: [ "1->2->5", "1->3" ]
     *
     * @param root
     *            the root of the binary tree
     * @return all root-to-leaf paths
     */
    @tags.BinaryTree
    @tags.BinaryTreeTraversal
    @tags.DFS
    @tags.Company.Facebook
    @tags.Company.Google
    public List<String> binaryTreePaths(TreeNode root) {
        if (root == null) {
            return Collections.emptyList();
        }
        List<String> result = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        binaryTreePaths(root, result, sb);
        return result;
    }

    private void binaryTreePaths(TreeNode node, List<String> result,
            StringBuilder sb) {
        int oldLen = sb.length();
        if (sb.length() != 0) {
            sb.append("->");
        }
        sb.append(node.val);

        if (node.left == null && node.right == null) {
            result.add(sb.toString());
        } else {
            if (node.left != null) {
                binaryTreePaths(node.left, result, sb);
            }
            if (node.right != null) {
                binaryTreePaths(node.right, result, sb);
            }
        }

        sb.delete(oldLen, sb.length());
    }

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
    @tags.Recursion
    @tags.DFS
    @tags.Tree
    @tags.BinaryTree
    @tags.DivideAndConquer
    @tags.Company.Apple
    @tags.Company.LinkedIn
    @tags.Company.Uber
    @tags.Company.Yahoo
    @tags.Status.Easy
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
     * Example: Given binary tree A = {3,9,20,#,#,15,7}, B = {3,#,20,15,7}. The
     * binary tree A is a height-balanced binary tree, but B is not.
     *
     * @param root:
     *            The root of binary tree.
     * @return: True if this Binary tree is Balanced, or false.
     */
    @tags.Recursion
    @tags.DivideAndConquer
    @tags.Tree
    @tags.DFS
    @tags.Company.Bloomberg
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
    @tags.DynamicProgramming
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
    public int maxPathSumII(TreeNode root) {
        if (root != null) {
            int leftMax = maxPathSumII(root.left);
            int rightMax = maxPathSumII(root.right);
            int childMax = Math.max(leftMax, rightMax);

            return root.val + Math.max(childMax, 0);
        }
        return 0;
    }

    /**
     * Binary Tree Path Sum.
     *
     * Given a binary tree, find all paths that sum of the nodes in the path
     * equals to a given number target.
     *
     * A valid path is from root node to any of the leaf nodes.
     *
     * @param root
     *            the root of binary tree
     * @param target
     *            an integer
     * @return all valid paths
     */
    @tags.BinaryTree
    @tags.BinaryTreeTraversal
    @tags.Status.Easy
    public List<List<Integer>> binaryTreePathSum(TreeNode root, int target) {
        List<List<Integer>> result = new ArrayList<>();
        if (root != null) {
            binaryTreePathSum(root, target, result, new ArrayList<Integer>());
        }

        return result;
    }

    private void binaryTreePathSum(TreeNode root, int target,
            List<List<Integer>> result, List<Integer> path) {
        target -= root.val;
        path.add(root.val);

        if (root.left == null && root.right == null && target == 0) {
            result.add(new ArrayList<>(path));
        }

        if (root.left != null) {
            binaryTreePathSum(root.left, target, result, path);
        }
        if (root.right != null) {
            binaryTreePathSum(root.right, target, result, path);
        }

        path.remove(path.size() - 1);
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
     * The left subtree of a node contains only nodes with keys less than the
     * node's key. The right subtree of a node contains only nodes with keys
     * greater than the node's key. Both the left and right subtrees must also
     * be binary search trees. A single node tree is a BST.
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
        return isValidBST(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
    }

    private boolean isValidBST(TreeNode node, int min, int max) {
        if (node == null) {
            return true;
        }

        if (min <= node.val && node.val <= max) {
            boolean left = isValidBST(node.left, min, node.val);
            boolean right = isValidBST(node.right, node.val, max);
            return left && right;
        }
        return false;
    }

    /**
     * Complete Binary Tree.
     *
     * Check a binary tree is completed or not. A complete binary tree is a
     * binary tree that every level is completed filled except the deepest
     * level. In the deepest level, all nodes must be as left as possible. See
     * more definition.
     *
     * @param root,
     *            the root of binary tree.
     * @return true if it is a complete binary tree, or false.
     */
    @tags.BinaryTree
    public boolean isComplete(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        boolean foundLeafLevel = false;
        while (!queue.isEmpty()) {
            for (int i = 0; i < queue.size(); i++) {
                TreeNode node = queue.poll();
                if (node != null && foundLeafLevel) {
                    return false;
                }
                if (node == null) {
                    foundLeafLevel = true;
                }
                if (node != null) {
                    queue.offer(node.left);
                    queue.offer(node.right);
                }
            }
        }

        return true;
    }

    /**
     * Minimum Depth of Binary Tree.
     *
     * Given a binary tree, find its minimum depth. The minimum depth is the
     * number of nodes along the shortest path from the root node down to the
     * nearest leaf node.
     *
     * @param root:
     *            The root of binary tree.
     * @return: An integer.
     */
    @tags.Tree
    @tags.BinaryTree
    @tags.DFS
    @tags.BFS
    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }

        int left = minDepth(root.left);
        int right = minDepth(root.right);

        if (left * right == 0) {
            return Math.max(left, right) + 1;
        } else {
            return Math.min(left, right) + 1;
        }
    }

    /**
     * Insert Node in a Binary Search Tree.
     *
     * Given a binary search tree and a new tree node, insert the node into the
     * tree. You should keep the tree still be a valid binary search tree.
     *
     * Notice: You can assume there is no duplicate values in this tree + node.
     *
     * Challenge: Can you do it without recursion?
     *
     * @param root:
     *            The root of the binary search tree.
     * @param node:
     *            insert this node into the binary search tree
     * @return: The root of the new binary search tree.
     */
    @tags.BinarySearchTree
    @tags.Source.LintCode
    public TreeNode insertNode(TreeNode root, TreeNode node) {
        if (root == null) {
            return node;
        }

        if (root.val > node.val) {
            root.left = insertNode(root.left, node);
        } else {
            root.right = insertNode(root.right, node);
        }

        return root;
    }

    /** Insert Node in a Binary Search Tree - iterative. */
    public TreeNode insertNode2(TreeNode root, TreeNode node) {
        if (root == null) {
            return node;
        }

        TreeNode current = root;
        while (true) {
            if (current.val < node.val) {
                if (current.right == null) {
                    current.right = node;
                    break;
                }
                current = current.right;
            } else {
                if (current.left == null) {
                    current.left = node;
                    break;
                }
                current = current.left;
            }
        }

        return root;
    }

    /**
     * Construct Binary Tree from Preorder and Inorder Traversal.
     *
     * Given preorder and inorder traversal of a tree, construct the binary
     * tree.
     *
     * Notice: You may assume that duplicates do not exist in the tree.
     *
     * @param preorder
     *            : A list of integers that preorder traversal of a tree
     * @param inorder
     *            : A list of integers that inorder traversal of a tree
     * @return : Root of a tree
     */
    @tags.BinaryTree
    @tags.Status.NeedPractice
    public TreeNode buildTreePreIn(int[] preorder, int[] inorder) {
        int len = preorder.length;
        return buildTreePreIn(preorder, 0, len - 1, inorder, 0, len - 1);
    }

    private TreeNode buildTreePreIn(int[] preorder, int preStart, int preEnd,
                              int[] inorder, int inStart, int inEnd) {
        if (inStart > inEnd) {
            return null;
        }

        TreeNode root = new TreeNode(preorder[preStart]);
        int index = inStart;
        for (; index <= inEnd; index++) {
            if (inorder[index] == root.val) {
                break;
            }
        }

        root.left = buildTreePreIn(preorder, preStart + 1,
                preStart + index - inStart, inorder, inStart, index - 1);
        root.right = buildTreePreIn(preorder, preStart + index - inStart + 1,
                preEnd, inorder, index + 1, inEnd);
        return root;
    }

    /**
     * Construct Binary Tree from Inorder and Postorder Traversal.
     *
     * Given inorder and postorder traversal of a tree, construct the binary
     * tree.
     *
     * Notice: You may assume that duplicates do not exist in the tree.
     *
     * @param inorder
     *            : A list of integers that inorder traversal of a tree
     * @param postorder
     *            : A list of integers that postorder traversal of a tree
     * @return : Root of a tree
     */
    @tags.BinaryTree
    @tags.Status.NeedPractice
    public TreeNode buildTreeInPost(int[] inorder, int[] postorder) {
        int len = postorder.length;
        return buildTreeInPost(postorder, 0, len - 1, inorder, 0, len - 1);
    }

    private TreeNode buildTreeInPost(int[] postorder, int postStart, int postEnd,
                              int[] inorder, int inStart, int inEnd) {
        if (inStart > inEnd) {
            return null;
        }

        TreeNode root = new TreeNode(postorder[postEnd]);
        int index = inStart;
        for (; index <= inEnd; index++) {
            if (inorder[index] == root.val) {
                break;
            }
        }

        root.left = buildTreeInPost(inorder, inStart, index - 1, postorder, postStart,
                postStart + index - inStart - 1);
        root.right = buildTreeInPost(inorder, index + 1, inEnd, postorder,
                postStart + index - inStart, postEnd - 1);

        return root;
    }

    /**
     * Search Range in Binary Search Tree.
     *
     * Given two values k1 and k2 (where k1 < k2) and a root pointer to a Binary
     * Search Tree. Find all the keys of tree in range k1 to k2. i.e. print all
     * x such that k1<=x<=k2 and x is a key of given BST. Return all the keys in
     * ascending order.
     *
     * @param root:
     *            The root of the binary search tree.
     * @param k1
     *            and k2: range k1 to k2.
     * @return: Return all keys that k1<=key<=k2 in ascending order.
     */
    @tags.BinaryTree
    @tags.BinarySearchTree
    public ArrayList<Integer> searchRange(TreeNode root, int k1, int k2) {
        ArrayList<Integer> result = new ArrayList<>();
        inorderTraverse(root, result, k1, k2);
        return result;
    }

    private void inorderTraverse(TreeNode node, ArrayList<Integer> result,
                                 int k1, int k2) {
        if (node != null) {
            inorderTraverse(node.left, result, k1, k2);
            if (node.val >= k1 && node.val <= k2) {
                result.add(node.val);
            }
            inorderTraverse(node.right, result, k1, k2);
        }
    }

    /** Search Range in Binary Search Tree - iterative solution. */
    public ArrayList<Integer> searchRange2(TreeNode root, int k1, int k2) {
        ArrayList<Integer> result = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode current = root;

        while (current != null || !stack.isEmpty()) {
            while (current != null) {
                stack.push(current);
                current = current.left;
            }

            TreeNode node = stack.pop();
            if (node != null) {
                if (k1 <= node.val && node.val <= k2) {
                    result.add(node.val);
                }
                current = node.right;
            }
        }

        return result;
    }

    /**
     * Remove Node in Binary Search Tree.
     *
     * Given a root of Binary Search Tree with unique value for each node.
     * Remove the node with given value. If there is no such a node with given
     * value in the binary search tree, do nothing. You should keep the tree
     * still a binary search tree after removal.
     *
     * @param root:
     *            The root of the binary search tree.
     * @param value:
     *            Remove the node with given value.
     * @return: The root of the binary search tree after removal.
     */
    @tags.BinarySearchTree
    @tags.Source.LintCode
    @tags.Status.Hard
    public TreeNode removeNode(TreeNode root, int value) {
        // find parent of the node to remove
        TreeNode prev = null;
        TreeNode current = root;
        while (current != null) {
            if (current.val < value) {
                prev = current;
                current = current.right;
            } else if (current.val > value) {
                prev = current;
                current = current.left;
            } else {
                break;
            }
        }

        // not found
        if (current == null) {
            return root;
        }

        // combine the left child and right child
        TreeNode left = current.left;
        TreeNode right = current.right;
        if (prev == null) {
            root = combine(left, right);
        } else {
            if (prev.left == current) {
                prev.left = combine(left, right);
            } else {
                prev.right = combine(left, right);
            }
        }

        return root;
    }

    private TreeNode combine(TreeNode n1, TreeNode n2) {
        if (n1 == null) {
            return n2;
        } else if (n2 == null) {
            return n1;
        }

        TreeNode current = n1;
        while (current.right != null) {
            current = current.right;
        }
        current.right = n2;

        return n1;
    }

    /**
     * Find Leaves of Binary Tree.
     *
     * Given a binary tree, find all leaves and then remove those leaves. Then
     * repeat the previous steps until the tree is empty.
     *
     * Example:
     * Given binary tree 
     *           1
     *          / \
     *         2   3
     *        / \     
     *       4   5    
     * Returns [4, 5, 3], [2], [1].
     */
    @tags.Tree
    @tags.DFS
    @tags.BinaryTree
    @tags.Source.LeetCode
    @tags.Company.LinkedIn
    @tags.Status.Easy
    public List<List<Integer>> findLeaves(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        height(root, result);
        return result;
    }

    private int height(TreeNode root, List<List<Integer>> result) {
        if (root == null) {
            return -1;
        }
        int height = Math.max(height(root.left, result),
                height(root.right, result)) + 1;
        if (height == result.size()) {
            result.add(new ArrayList<Integer>());
        }
        result.get(height).add(root.val);
        return height;
    }

    /**
     * Binary Search Tree Iterator.
     *
     * Design an iterator over a binary search tree with the following rules: 1.
     * Elements are visited in ascending order (i.e. an in-order traversal). 2.
     * next() and hasNext() queries run in O(1) time in average.
     */
    @tags.Design
    @tags.Stack
    @tags.Tree
    @tags.BinaryTree
    @tags.NonRecursion
    @tags.BinarySearchTree
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Company.LinkedIn
    @tags.Company.Microsoft
    @tags.Status.OK
    class BSTIterator {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode current;

        /** @param root: The root of binary tree. */
        public BSTIterator(TreeNode root) {
            current = root;
        }

        /** @return: true if the iteration has more elements. */
        public boolean hasNext() {
            while (current != null) {
                stack.push(current);
                current = current.left;
            }
            return !stack.isEmpty();
        }

        /** @return: the next element in the iteration. */
        public TreeNode next() {
            if (hasNext()) {
                TreeNode next = stack.pop();
                current = next.right;
                return next;
            }
            throw new NoSuchElementException();
        }
    }

    /**
     * Unique Binary Search Trees.
     *
     * Given n, how many structurally unique BSTs (binary search trees) that
     * store values 1...n?
     *
     * Example: Given n = 3, there are a total of 5 unique BST's.
     *
     * @paramn n: An integer
     * @return: An integer
     */
    @tags.DynamicProgramming
    @tags.CatalanNumber
    public int numTrees(int n) {
        int[] treeCount = new int[n + 1];
        treeCount[0] = 1;

        for (int i = 1; i <= n; i++) { // fill dp array increasingly
            for (int j = 1; j <= i; j++) { // root can be any node
                treeCount[i] += treeCount[j - 1] * treeCount[i - j];
            }
        }

        return treeCount[n];
    }

    /** Unique Binary Search Trees - O(1) space. */
    public int numTrees2(int n) {
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
     * Unique Binary Search Trees II.
     *
     * Given n, generate all structurally unique BST's (binary search trees)
     * that store values 1...n.
     *
     * Example: Given n = 3, your program should return all 5 unique BSTs.
     *
     * @paramn n: An integer
     * @return: A list of root
     */
    @tags.DFS
    @tags.DynamicProgramming
    public List<TreeNode> generateTrees(int n) {
        return generateTrees(1, n);
    }

    private List<TreeNode> generateTrees(int start, int end) {
        List<TreeNode> trees = new ArrayList<>();
        if (start > end) {
            trees.add(null); // Notice: we need a null to represent empty tree
            return trees;
        }

        for (int i = start; i <= end; i++) {
            List<TreeNode> left = generateTrees(start, i - 1);
            List<TreeNode> right = generateTrees(i + 1, end);
            for (TreeNode ln : left) {
                for (TreeNode rn : right) {
                    TreeNode root = new TreeNode(i);
                    root.left = ln;
                    root.right = rn;
                    trees.add(root);
                }
            }
        }

        return trees;
    }

    /**
     * Closest Binary Search Tree Value.
     *
     * Given a non-empty binary search tree and a target value, find the value
     * in the BST that is closest to the target.
     *
     * Note: Given target value is a floating point. You are guaranteed to have
     * only one unique value in the BST that is closest to the target.
     *
     * @param root
     * @param target
     * @return
     */
    @tags.Tree
    @tags.BinarySearch
    @tags.Company.Google
    @tags.Company.Microsoft
    @tags.Company.Snapchat
    @tags.Status.NeedPractice
    public int closestValue(TreeNode root, double target) {
        double diff = Math.abs(root.val - target);
        int closest = root.val;
        while (root != null) {
            if (root.val == target) {
                return root.val;
            }

            double newDiff = Math.abs(root.val - target);
            if (newDiff < diff) {
                diff = newDiff;
                closest = root.val;
            }

            if (root.val > target) {
                root = root.left;
            } else if (root.val < target) {
                root = root.right;
            }
        }

        return closest;
    }

    /**
     * Binary Tree Upside Down.
     *
     * Given a binary tree where all the right nodes are either leaf nodes with
     * a sibling (a left node that shares the same parent node) or empty, flip
     * it upside down and turn it into a tree where the original right nodes
     * turned into left leaf nodes. Return the new root.
     *
     * For example: Given a binary tree {1,2,3,4,5}, return the root of the
     * binary tree [4,5,2,#,#,3,1].
     *
     * @param root
     * @return
     */
    @tags.Tree
    @tags.Company.LinkedIn
    public TreeNode upsideDownBinaryTree(TreeNode root) {
        if (root == null || (root.left == null && root.right == null)) {
            return root;
        }

        TreeNode left = upsideDownBinaryTree(root.left);
        root.left.left = root.right;
        root.left.right = root;
        root.left = root.right = null;

        return left;
    }

    /**
     * Factor Combinations.
     *
     * Numbers can be regarded as product of its factors. For example, 8 = 2 x 2
     * x 2 = 2 x 4. Write a function that takes an integer n and return all
     * possible combinations of its factors.
     *
     * Note: You may assume that n is always positive. Factors should be greater
     * than 1 and less than n.
     *
     * Examples: input: 1 output: []. input: 37 output: []. input: 12 output:
     * [[2,6],[2,2,3],[3,4]]. input: 32 output:
     * [[2,16],[2,2,8],[2,2,2,4],[2,2,2,2,2],[2,4,4],[4,8]].
     *
     * @param n
     * @return
     */
    @tags.Backtracking
    @tags.Company.LinkedIn
    @tags.Company.Uber
    public List<List<Integer>> getFactors(int n) {
        List<List<Integer>> result = new ArrayList<>();
        getFactors(n, result, new ArrayList<Integer>());
        return result;
    }

    private void getFactors(int n, List<List<Integer>> result, List<Integer> path) {
        if (n == 1) {
            if (path.size() > 1) {
                result.add(new ArrayList<>(path));
            }
            return;
        }

        int last = path.isEmpty() ? 2 : path.get(path.size() - 1);
        for (int i = last; i <= n; i++) {
            if (n % i == 0) {
                path.add(i);
                getFactors(n / i, result, path);
                path.remove(path.size() - 1);
            }
        }
    }

    // ---------------------------------------------------------------------- //
    // ----------------------- Lowest Common Ancestor ----------------------- //
    // ---------------------------------------------------------------------- //

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
    @tags.Tree
    @tags.BinaryTree
    @tags.Company.Amazon
    @tags.Company.Apple
    @tags.Company.Facebook
    @tags.Company.LinkedIn
    @tags.Company.Microsoft
    @tags.Status.NeedPractice
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

    /**
     * Lowest Common Ancestor II.
     *
     * Given the root and two nodes in a Binary Tree. Find the lowest common
     * ancestor(LCA) of the two nodes.
     *
     * The lowest common ancestor is the node with largest depth which is the
     * ancestor of both nodes.
     *
     * The node has an extra attribute parent which point to the father of
     * itself. The root's parent is null.
     *
     * @param root:
     *            The root of the tree
     * @param A,
     *            B: Two node in the tree
     * @return: The lowest common ancestor of A and B
     */
    @tags.BinaryTree
    @tags.Source.LintCode
    public ParentTreeNode lowestCommonAncestorII(ParentTreeNode root,
                                                 ParentTreeNode A,
                                                 ParentTreeNode B) {
        int heightA = getHeight(A);
        int heightB = getHeight(B);

        while (heightA > heightB) {
            A = A.parent;
            heightA--;
        }

        while (heightB > heightA) {
            B = B.parent;
            heightB--;
        }

        while (A != B) {
            A = A.parent;
            B = B.parent;
        }

        return A;
    }

    private int getHeight(ParentTreeNode node) {
        int height = 0;
        while (node != null) {
            node = node.parent;
            height++;
        }
        return height;
    }

    // ---------------------------------------------------------------------- //
    // ------------------------- Next Right Pointer ------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Populating Next Right Pointers in Each Node.
     *
     * Given a binary tree
     *
     * struct TreeLinkNode { TreeLinkNode *left; TreeLinkNode *right;
     * TreeLinkNode *next; } Populate each next pointer to point to its next
     * right node. If there is no next right node, the next pointer should be
     * set to NULL.
     *
     * Initially, all next pointers are set to NULL.
     *
     * Note:
     * You may only use constant extra space. You may assume that it is a
     * perfect binary tree (ie, all leaves are at the same level, and every
     * parent has two children).
     *
     * For example, Given the following perfect binary tree,
     *        1
     *      /  \
     *     2    3
     *    / \  / \
     *   4  5  6  7
     *
     * After calling your function, the tree should look like:
     *
     *        1 -> NULL
     *      /  \
     *     2 -> 3 -> NULL
     *    / \  / \
     *   4->5->6->7 -> NULL
     *
     * @param root
     */
    @tags.DFS
    @tags.Tree
    @tags.Company.Microsoft
    @tags.Status.NeedPractice
    public void connect(TreeLinkNode root) {
        if (root == null) {
            return;
        }

        // connect left and right
        if (root.left != null) {
            root.left.next = root.right;
        }

        // connect right to next using root.next
        if (root.right != null && root.next != null) {
            root.right.next = root.next.left;
        }

        connect(root.left);
        connect(root.right);
    }

    /**
     * Populating Next Right Pointers in Each Node II.
     *
     * Follow up for problem "Populating Next Right Pointers in Each Node". What
     * if the given tree could be any binary tree? Would your previous solution
     * still work?
     *
     * Note: You may only use constant extra space.
     *
     * For example, Given the following binary tree,
     *         1
     *       /  \
     *      2    3
     *     / \    \
     *    4   5    7
     * After calling your function, the tree should look like:
     *         1 -> NULL
     *       /  \
     *      2 -> 3 -> NULL
     *     / \    \
     *    4-> 5 -> 7 -> NULL
     *
     * @param root
     */
    @tags.Tree
    @tags.DFS
    @tags.Company.Bloomberg
    @tags.Company.Facebook
    @tags.Company.Microsoft
    @tags.Status.NeedPractice
    public void connectII(TreeLinkNode root) {
        // can be solved with tail recursion

        TreeLinkNode lastLevel = root;

        while (lastLevel != null) {
            TreeLinkNode dummy = new TreeLinkNode(0);
            TreeLinkNode prev = dummy;

            // traverse last level by next pointer
            while (lastLevel != null) {
                if (lastLevel.left != null) {
                    prev.next = lastLevel.left;
                    prev = prev.next;
                }
                if (lastLevel.right != null) {
                    prev.next = lastLevel.right;
                    prev = prev.next;
                }

                lastLevel = lastLevel.next;
            }

            lastLevel = dummy.next;
        }
    }

    // ---------------------------------------------------------------------- //
    // --------------------- Binary Tree Serialization ---------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Binary Tree Serialization.
     *
     * Design an algorithm and write code to serialize and deserialize a binary
     * tree. Writing the tree to a file is called 'serialization' and reading
     * back from the file to reconstruct the exact same binary tree is
     * 'deserialization'.
     *
     * There is no limit of how you deserialize or serialize a binary tree, you
     * only need to make sure you can serialize a binary tree to a string and
     * deserialize this string to the original structure.
     */
    @tags.Tree
    @tags.Design
    @tags.BinaryTree
    @tags.Company.Amazon
    @tags.Company.Bloomberg
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Company.LinkedIn
    @tags.Company.Microsoft
    @tags.Company.Uber
    @tags.Company.Yahoo
    @tags.Status.Hard
    public class Codec {

        /**
         * This method will be invoked first, you should design your own
         * algorithm to serialize a binary tree which denote by a root node to a
         * string which can be easily deserialized by your own "deserialize"
         * method later.
         */
        public String serialize(TreeNode root) {
            Queue<TreeNode> queue = new LinkedList<>();
            queue.offer(root);
            StringBuilder sb = new StringBuilder();

            while (!queue.isEmpty()) {
                for (int i = 0; i < queue.size(); i++) {
                    TreeNode node = queue.poll();
                    if (node == null) {
                        sb.append('#');
                    } else {
                        sb.append(node.val);
                        queue.offer(node.left);
                        queue.offer(node.right);
                    }
                    sb.append(',');
                }
            }

            return sb.substring(0, sb.length() - 1);
        }

        /**
         * This method will be invoked second, the argument data is what exactly
         * you serialized at method "serialize", that means the data is not
         * given by system, it's given by your own serialize method. So the
         * format of data is designed by yourself, and deserialize it here as
         * you serialize it in "serialize" method.
         */
        public TreeNode deserialize(String data) {
            String[] nodes = data.split(",");
            int index = 0;

            TreeNode root = null;
            if (!nodes[index].equals("#")) {
                root = new TreeNode(Integer.parseInt(nodes[index]));
            }
            index++;

            Queue<TreeNode> queue = new LinkedList<>();
            queue.offer(root);

            while (!queue.isEmpty()) {
                TreeNode node = queue.poll();
                if (node != null) {
                    if (!nodes[index].equals("#")) {
                        node.left = new TreeNode(
                                Integer.parseInt(nodes[index]));
                    }
                    index++;
                    if (!nodes[index].equals("#")) {
                        node.right = new TreeNode(
                                Integer.parseInt(nodes[index]));
                    }
                    index++;
                    queue.offer(node.left);
                    queue.offer(node.right);
                }
            }

            return root;
        }
    }

    /** Binary Tree Serialization - DFS solution. */
    @tags.BinaryTree
    @tags.Company.Microsoft
    @tags.Company.Yahoo
    @tags.Status.Hard

    public String serializeDFS(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);

        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            if (node == null) {
                sb.append('#');
            } else {
                sb.append(node.val);
                stack.push(node.right);
                stack.push(node.left);
            }
            sb.append(',');
        }

        return sb.substring(0, sb.length() - 1);
    }

    public TreeNode deserializeDFS(String data) {
        String[] nodes = data.split(",");
        TreeNode root = null;
        if (!nodes[0].equals("#")) {
            root = new TreeNode(Integer.parseInt(nodes[0]));
        }

        Stack<TreeNode> stack = new Stack<>();
        TreeNode current = root;
        for (int i = 1; i < nodes.length; i++) {
            TreeNode next = nodes[i].equals("#") ? null
                    : new TreeNode(Integer.parseInt(nodes[i]));
            if (current != null) {
                current.left = next;
                stack.push(current);
                current = next;
            } else {
                stack.peek().right = next;
                stack.pop();
                current = next;
            }
        }
        return root;
    }

    // ---------------------------------------------------------------------- //
    // ---------------------- Same, Symmetric, Tweaked ---------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Same Tree (Identical Binary Tree).
     *
     * Given two binary trees, write a function to check if they are equal or
     * not. Two binary trees are considered equal if they are structurally
     * identical and the nodes have the same value.
     *
     * @param a,b
     *            the root of binary trees.
     * @return true if they are identical, or false.
     */
    @tags.Tree
    @tags.BinaryTree
    @tags.DFS
    @tags.Recursion
    @tags.Company.Bloomberg
    @tags.Status.Easy
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        } else if (p == null || q == null) {
            return false;
        }

        return p.val == q.val && isSameTree(p.left, q.left)
                && isSameTree(p.right, q.right);
    }

    /**
     * Subtree.
     *
     * You have two every large binary trees: T1, with millions of nodes, and
     * T2, with hundreds of nodes. Create an algorithm to decide if T2 is a
     * subtree of T1.
     *
     * Notice: A tree T2 is a subtree of T1 if there exists a node n in T1 such
     * that the subtree of n is identical to T2. That is, if you cut off the
     * tree at node n, the two trees would be identical.
     *
     * @param T1,
     *            T2: The roots of binary tree.
     * @return: True if T2 is a subtree of T1, or false.
     */
    @tags.BinaryTree
    @tags.Recursion
    @tags.Status.NeedPractice
    public boolean isSubtree(TreeNode T1, TreeNode T2) {
        if (isSameTree(T1, T2)) {
            return true;
        }
        if (T1 != null) {
            return isSubtree(T1.left, T2) || isSubtree(T1.right, T2);
        }
        return false;
    }

    /**
     * Tweaked Identical Binary Tree.
     *
     * Check two given binary trees are identical or not. Assuming any number of
     * tweaks are allowed. A tweak is defined as a swap of the children of one
     * node in the tree.
     *
     * Notice: There is no two nodes with the same value in the tree.
     *
     * @param a, b, the root of binary trees.
     * @return true if they are tweaked identical, or false.
     */
    @tags.BinaryTree
    public boolean isTweakedIdentical(TreeNode a, TreeNode b) {
        if (a == null && b == null) {
            return true;
        }
        if (a == null || b == null || a.val != b.val) {
            return false;
        }

        if ((isTweakedIdentical(a.left, b.left) && isTweakedIdentical(a.right, b.right))
            || (isTweakedIdentical(a.left, b.right) && isTweakedIdentical(a.right, b.left))) {
            return true;
        }
        return false;
    }

    /**
     * Symmetric Binary Tree.
     *
     * Given a binary tree, check whether it is a mirror of itself (i.e.,
     * symmetric around its center).
     *
     * @param root,
     *            the root of binary tree.
     * @return true if it is a mirror of itself, or false.
     */
    @tags.Tree
    @tags.BinaryTree
    @tags.DFS
    @tags.Company.Bloomberg
    @tags.Company.LinkedIn
    @tags.Company.Microsoft
    @tags.Status.NeedPractice
    public boolean isSymmetricRec(TreeNode root) {
        if (root == null) {
            return true;
        }

        return isTweaked(root.left, root.right);
    }

    private boolean isTweaked(TreeNode a, TreeNode b) {
        if (a == null && b == null) {
            return true;
        }
        if (a == null || b == null || a.val != b.val) {
            return false;
        }

        return isTweaked(a.left, b.right) && isTweaked(a.right, b.left);
    }

    /** Symmetric Binary Tree - iterative solution. */
    @tags.Tree
    @tags.BinaryTree
    @tags.BFS
    @tags.Company.Bloomberg
    @tags.Company.LinkedIn
    @tags.Company.Microsoft
    @tags.Status.NeedPractice
    public boolean isSymmetricIter(TreeNode root) {
        if (root == null) return true;

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root.left);
        queue.offer(root.right);

        while (!queue.isEmpty()) {
            TreeNode left = queue.poll();
            TreeNode right = queue.poll();

            if (left == null && right == null) {
                continue;
            } else if (left == null || right == null || left.val != right.val) {
                return false;
            }

            queue.offer(left.left);
            queue.offer(right.right);
            queue.offer(left.right);
            queue.offer(right.left);
        }

        return true;
    }

    // ------------------------------ OLD ------------------------------------

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

    // ---------------------------------------------------------------------- //
    // ----------------------------- UNIT TESTS ----------------------------- //
    // ---------------------------------------------------------------------- //

    @Test
    public void test() {
        TreeNode n1 = null;
        TreeNode n2 = null;
        System.out.println(n1 != n2);

        int a = Integer.MIN_VALUE;
        System.out.println(a);
        int b = Integer.MIN_VALUE;
        int c = (a + b) >>> 1;
        System.out.println(c);

        levelOrderTest();
    }

    private void levelOrderTest() {
        TreeNode root = new TreeNode(1);
        TreeNode left = new TreeNode(1);
        TreeNode right = new TreeNode(3);
        root.left = left;
        root.right = right;

        levelOrder(root);
    }
}
