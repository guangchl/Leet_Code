package categories;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Stack;

import categories.BinaryTreeAndDivideConquer.TreeNode;

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

    /** Definition for binary tree with parent pointer */
    class ParentTreeNode {
        public ParentTreeNode parent, left, right;
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
     * @param root: The root of binary tree.
     * @return: Postorder in ArrayList which contains node values.
     */
    @tags.Recursion
    @tags.BinaryTree
    @tags.BinaryTreeTraversal
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

            current = stack.peek();
            if (current.right == null || current.right == prev) {
                stack.pop();
                result.add(current.val);
                prev = current;
                current = null;
            } else {
                current = current.right;
            }
        }

        return result;
    }

    /**
     * Binary Tree Inorder Traversal.
     *
     * Given a binary tree, return the inorder traversal of its nodes' values.
     *
     * @param root: The root of binary tree.
     * @return: Inorder in ArrayList which contains node values.
     */
    @tags.Recursion
    @tags.BinaryTree
    @tags.BinaryTreeTraversal
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

    /**
     * Binary Tree Zigzag Level Order Traversal.
     *
     * Given a binary tree, return the zigzag level order traversal of its
     * nodes' values. (ie, from left to right, then right to left for the next
     * level and alternate between).
     *
     * @param root:
     *            The root of binary tree.
     * @return: A list of lists of integer include the zigzag level order
     *          traversal of its nodes' values
     */
    @tags.Queue
    @tags.BFS
    @tags.BinaryTree
    @tags.BinaryTreeTraversal
    @tags.Company.LinkedIn
    public ArrayList<ArrayList<Integer>> zigzagLevelOrder(TreeNode root) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        Stack<TreeNode> thisLevel = new Stack<>();
        thisLevel.push(root);
        boolean normalOrder = true;

        while (!thisLevel.isEmpty()) {
            Stack<TreeNode> nextLevel = new Stack<>();
            ArrayList<Integer> level = new ArrayList<>();

            while (!thisLevel.isEmpty()) {
                TreeNode node = thisLevel.pop();
                if (node == null) continue;
                level.add(node.val);

                // push child to next level
                if (normalOrder) {
                    nextLevel.push(node.left);
                    nextLevel.push(node.right);
                } else {
                    nextLevel.push(node.right);
                    nextLevel.push(node.left);
                }
            }

            // append the result
            if (!level.isEmpty()) {
                thisLevel = nextLevel;
                result.add(level);
                normalOrder ^= true;
            }
        }

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
    public List<List<Integer>> binaryTreePathSum(TreeNode root, int target) {
        List<List<Integer>> result = new ArrayList<>();
        dfsBTPS(result, root, new ArrayList<Integer>(), 0, target);
        return result;
    }

    private void dfsBTPS(List<List<Integer>> result, TreeNode node, List<Integer> path, int sum, int target) {
        if (node != null) {
            sum += node.val;
            path.add(node.val);

            if (node.left == null && node.right == null) {
                if (sum == target) {
                    result.add(new ArrayList<>(path));
                }
            }
            if (node.left != null) {
                dfsBTPS(result, node.left, path, sum, target);
            }
            if (node.right != null) {
                dfsBTPS(result, node.right, path, sum, target);
            }
            path.remove(path.size() - 1);
        }
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
     * Identical Binary Tree (Same Tree).
     *
     * Check if two binary trees are identical. Identical means the two binary
     * trees have the same structure and every identical position has the same
     * value.
     *
     * @param a,
     *            b, the root of binary trees.
     * @return true if they are identical, or false.
     */
    @tags.BinaryTree
    public boolean isIdentical(TreeNode a, TreeNode b) {
        if (a == null && b == null) {
            return true;
        }

        if (a == null || b == null || a.val != b.val) {
            return false;
        }

        return isIdentical(a.left, b.left) && isIdentical(a.right, b.right);
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
    @tags.BinaryTree
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

    /**
     * Symmetric Binary Tree - iterative solution.
     */
    @tags.BinaryTree
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
    @tags.BinaryTree
    @tags.DFS
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
    public TreeNode buildTreePreIn(int[] preorder, int[] inorder) {
        int len = preorder.length;
        if (len != inorder.length) {
            return null;
        }

        return buildTreePreIn(preorder, 0, len - 1, inorder, 0, len - 1);
    }

    private TreeNode buildTreePreIn(int[] preorder, int preStart, int preEnd,
                              int[] inorder, int inStart, int inEnd) {
        if (inStart > inEnd) {
            return null;
        }

        int rootVal = preorder[preStart];
        TreeNode root = new TreeNode(rootVal);
        int index;
        for (index = inStart; index <= inEnd; index++) {
            if (inorder[index] == rootVal) {
                break;
            }
        }

        root.left = buildTreePreIn(preorder, preStart + 1, preStart + index - inStart, inorder, inStart, index - 1);
        root.right = buildTreePreIn(preorder, preStart + index - inStart + 1, preEnd, inorder, index + 1, inEnd);
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
    public TreeNode buildTreeInPost(int[] inorder, int[] postorder) {
        int len = postorder.length;
        if (inorder.length != len) {
            return null;
        }

        return buildTreeInPost(postorder, 0, len - 1, inorder, 0, len - 1);
    }

    private TreeNode buildTreeInPost(int[] postorder, int postStart, int postEnd,
                              int[] inorder, int inStart, int inEnd) {
        if (inStart > inEnd) {
            return null;
        }

        int rootVal = postorder[postEnd];
        TreeNode root = new TreeNode(rootVal);

        int index;
        for (index = inStart; index <= inEnd; index++) {
            if (inorder[index] == rootVal) {
                break;
            }
        }

        root.left = buildTreeInPost(postorder, postStart, postStart + index - inStart - 1, inorder, inStart, index - 1);
        root.right = buildTreeInPost(postorder, postStart + index - inStart, postEnd - 1, inorder, index + 1, inEnd);

        return root;
    }

    /**
     * Search Range in Binary Search Tree
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
    @tags.BinaryTree
    @tags.Company.Microsoft
    @tags.Company.Yahoo

    /**
     * This method will be invoked first, you should design your own algorithm 
     * to serialize a binary tree which denote by a root node to a string which
     * can be easily deserialized by your own "deserialize" method later.
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
     * This method will be invoked second, the argument data is what exactly you
     * serialized at method "serialize", that means the data is not given by
     * system, it's given by your own serialize method. So the format of data is
     * designed by yourself, and deserialize it here as you serialize it in
     * "serialize" method.
     */
    public TreeNode deserialize(String data) {
        String[] nodes = data.split(",");
        int index = 0;

        TreeNode root = null;
        if (!nodes[index].equals("#")) {
            root = new TreeNode(Integer.valueOf(nodes[index]));
        }
        index++;

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            for (int i = 0; i < queue.size(); i++) {
                TreeNode node = queue.poll();
                if (node == null) {
                    continue;
                }

                if (!nodes[index].equals("#")) {
                    node.left = new TreeNode(Integer.valueOf(nodes[index]));
                }
                index++;
                if (!nodes[index].equals("#")) {
                    node.right = new TreeNode(Integer.valueOf(nodes[index]));
                }
                index++;
                queue.offer(node.left);
                queue.offer(node.right);
            }
        }

        return root;
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
    public TreeNode removeNode(TreeNode root, int value) {
        TreeNode dummy = new TreeNode(0);
        dummy.right = root;

        // find the the parent of the node to remove
        TreeNode parent = findNode(dummy, root, value);

        // remove the node
        if (parent.left != null && parent.left.val == value) {
            deleteNode(parent, parent.left);
        } else if (parent.right != null && parent.right.val == value) {
            deleteNode(parent, parent.right);
        }

        return dummy.right;
    }

    private TreeNode findNode(TreeNode parent, TreeNode node, int value) {
        if (node == null || node.val == value) {
            return parent;
        } else if (node.val > value) {
            return findNode(node, node.left, value);
        } else {
            return findNode(node, node.right, value);
        }
    }

    private void deleteNode(TreeNode parent, TreeNode node) {
        TreeNode left = node.left;
        TreeNode right = node.right;
        TreeNode temp = null;

        if (right != null) {
            temp = right;
            while (temp.left != null) {
                temp = temp.left;
            }

            temp.left = left;
            temp = right;
        } else {
            temp = left;
        }

        if (parent.left == node) {
            parent.left = temp;
        } else {
            parent.right = temp;
        }
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
    @tags.DFS
    @tags.BinaryTree
    @tags.Source.LeetCode
    public List<List<Integer>> findLeaves(TreeNode root) {
        // TODO: locked problem from LeetCode
        return null;
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

    public void test () {
        TreeNode root = new TreeNode(1);
        TreeNode left = new TreeNode(1);
        TreeNode right = new TreeNode(3);
        root.left = left;
        root.right = right;

        levelOrder(root);

        TreeNode n1 = null;
        TreeNode n2 = null;
        System.out.println(n1 != n2);

        int a = Integer.MIN_VALUE;
        System.out.println(a);
        int b = Integer.MIN_VALUE;
        int c = (a + b) >>> 1;
        System.out.println(c);
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
