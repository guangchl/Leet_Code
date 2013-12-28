package leet_code;

import java.util.ArrayList;
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
//      int ones = 0, twos = 0, threes = 0;
//      for (int i = 0; i < n; i++) {
//          twos |= ones & A[i];
//          ones ^= A[i];
//          threes = ones & twos;
//          ones &= ~threes;
//          twos &= ~threes;
//      }
//      return ones;
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
     * Given a binary tree, return the post-order traversal of its nodes' values.
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
        int n = A[0];
        for (int i = 1; i < size; i++) {
            if (A[i] == n) {
                int end; // end index of the same duplicate
                for (end = i; end + 1 < size && A[end + 1] == n;) {
                    end++;
                }
                int len = end - i + 1; // length of this set of duplicates
                
                // left shift the part at the right of the set of duplicates
                for (int j = i; j + len < size; j++) {
                    A[j] = A[j + len]; 
                }
                size -= len;
            }
            
            // set a new value to find duplicates with
            n = A[i];
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
        
        ArrayList<TreeNode> level = new ArrayList<TreeNode>();
        level.add(root);
        
        while (!level.isEmpty()) {
            int size = level.size();
            
            for (int i = 0; i < size / 2; i++) {
                if ((level.get(i) == null && level.get(size - i - 1) != null)
                        || (level.get(i) != null && level.get(size - i - 1) == null)) {
                    return false;
                } else if (level.get(i) != null && level.get(size - i - 1) != null) {
                    if (level.get(i).val != level.get(size - i - 1).val) {
                        return false;
                    }
                }
            }
            
            ArrayList<TreeNode> newLevel = new ArrayList<TreeNode>(size * 2);
            for (int i = 0; i < size; i++) {
                if (level.get(i) != null) {
                    newLevel.add(level.get(i).left);
                    newLevel.add(level.get(i).right);
                }
            }
            level = newLevel;
        }
        
        return true;
    }
    
    public boolean isSymmetricRecursive(TreeNode root) {
        if (root == null) {
            return true;
        }
        
        isSymmetricRecursive(root.left) = isSymmetricRecursive(root.right);
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
        
    }
    
    public TreeNode sortedArrayToBST(int[] num, int start, int end) {
        
    }

    /**
     * Roman to Integer
     * 
     * Given a Roman numeral, convert it to an integer.
     * 
     * Input is guaranteed to be within the range from 1 to 3999.
     */
    public int romanToInt(String s) {
        
    }
    
    /**
     * Integer to Roman
     * 
     * Given an integer, convert it to a roman numeral.
     * 
     * Input is guaranteed to be within the range from 1 to 3999.
     */
    public String intToRoman(int num) {
        
    }
    
    public void test() {
        //int[] A = { 1, 2, 3, 3, 4, 4, 5 };
        TreeNode node = new TreeNode(1);
        System.out.println(isSymmetricIterative(node));
    }


    public static void main(String[] args) {
        Problems m = new Problems();
        m.test();
    }

}
