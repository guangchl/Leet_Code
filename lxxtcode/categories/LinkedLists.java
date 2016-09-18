package categories;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Set;
import java.util.Stack;

public class LinkedLists {

    // ---------------------------------------------------------------------- //
    // ------------------------------- MODELS ------------------------------- //
    // ---------------------------------------------------------------------- //

    /** Definition for singly-linked list. */
    public class ListNode {
        int val;
        ListNode next;

        public ListNode(int x) {
            val = x;
            next = null;
        }
    }

    /** Definition of TreeNode. */
    public class TreeNode {
        public int val;
        public TreeNode left, right;

        public TreeNode(int val) {
            this.val = val;
            this.left = this.right = null;
        }
    }

    /** Definition for singly-linked list with a random pointer. */
    public class RandomListNode {
        int label;
        RandomListNode next, random;

        RandomListNode(int x) {
            this.label = x;
        }
    }

    /** Definition for Doubly-ListNode. */
    public class DoublyListNode {
        int val;
        DoublyListNode next, prev;

        DoublyListNode(int val) {
            this.val = val;
            this.next = this.prev = null;
        }
    }

    // ---------------------------------------------------------------------- //
    // ------------------------------ TEMPLATE ------------------------------ //
    // ---------------------------------------------------------------------- //

    /**
     * Remove a node from a single linked list
     */
    public boolean remove(ListNode head, int val) {
        if (head == null) {
            return false;
        }

        ListNode dummy = new ListNode(0);
        dummy.next = head;
        head = dummy;

        while (head.next != null) {
            if (head.next.val == val) {
                head.next = head.next.next;
                return true;
            }

            head = head.next;
        }

        return false;
    }

    /**
     * Reverse Linked List.
     *
     * Reverse a single linked list - iterative.
     *
     * Example For linked list 1->2->3, the reversed linked list is 3->2->1
     *
     * @param head:
     *            The head of linked list.
     * @return: The new head of reversed linked list.
     */
    @tags.LinkedList
    @tags.Source.LeetCode
    @tags.Source.LintCode
    @tags.Company.Adobe
    @tags.Company.Amazon
    @tags.Company.Apple
    @tags.Company.Bloomberg
    @tags.Company.Facebook
    @tags.Company.Microsoft
    @tags.Company.Snapchat
    @tags.Company.Twitter
    @tags.Company.Uber
    @tags.Company.Yahoo
    @tags.Company.Yelp
    @tags.Company.Zenefits
    @tags.Status.Easy
    public ListNode reverse(ListNode head) {
        ListNode newHead = null;

        while (head != null) {
            ListNode next = head.next;
            head.next = newHead;
            newHead = head;
            head = next;
        }

        return newHead;
    }

    /**
     * Reverse Linked List.
     *
     * Reverse a single linked list - recursive.
     */
    @tags.LinkedList
    @tags.Recursion
    @tags.Source.LeetCode
    @tags.Source.LintCode
    @tags.Company.Facebook
    @tags.Company.Uber
    @tags.Status.Easy
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode newHead = reverseList(head.next);
        head.next.next = head;
        head.next = null;

        return newHead;
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
     *
     * @param ListNode
     *            head is the head of the linked list
     * @oaram m and n
     * @return: The head of the reversed ListNode
     */
    @tags.LinkedList
    @tags.Status.NeedPractice
    public ListNode reverseBetween(ListNode head, int m, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;

        ListNode leftTail = dummy;
        while (--m > 0 && leftTail != null) {
            leftTail = leftTail.next;
            n--;
        }

        if (leftTail == null || leftTail.next == null) {
            return head;
        }

        ListNode midTail = leftTail.next;
        while (--n > 0 && midTail.next != null) {
            ListNode next = leftTail.next;
            leftTail.next = midTail.next;
            midTail.next = midTail.next.next;
            leftTail.next.next = next;
        }

        return dummy.next;
    }

    /**
     * Middle of Linked List
     *
     * Find the middle node of a linked list.
     *
     * Example: Given 1->2->3, return the node with value 2. Given 1->2, return
     * the node with value 1.
     *
     * @param head:
     *            the head of linked list.
     * @return: a middle node of the linked list
     */
    @tags.LinkedList
    public ListNode middleNode(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode fast = head.next, slow = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    // ---------------------------------------------------------------------- //
    // ------------------------------ PROBLEMS ------------------------------ //
    // ---------------------------------------------------------------------- //

    /**
     * Nth to Last Node in List.
     *
     * Find the nth to last element of a singly linked list. The minimum number
     * of nodes in list is n.
     *
     * Example: Given a List 3->2->1->5->null and n = 2, return node whose value
     * is 1.
     *
     * @param head:
     *            The first node of linked list.
     * @param n:
     *            An integer.
     * @return: Nth to last node of a singly linked list.
     */
    @tags.LinkedList
    @tags.TwoPointers
    @tags.Source.CrackingTheCodingInterview
    public ListNode nthToLast(ListNode head, int n) {
        ListNode current = head;
        while (n-- > 0) {
            current = current.next;
        }
        while (current != null) {
            current = current.next;
            head = head.next;
        }
        return head;
    }

    /**
     * Remove Nth Node From End of List.
     *
     * Given a linked list, remove the nth node from the end of list and return
     * its head.
     *
     * Notice: The minimum number of nodes in list is n.
     *
     * Example: Given linked list: 1->2->3->4->5->null, and n = 2. After
     * removing the second node from the end, the linked list becomes
     * 1->2->3->5->null.
     *
     * Challenge: Can you do it without getting the length of the linked list?
     *
     * @param head:
     *            The first node of linked list.
     * @param n:
     *            An integer.
     * @return: The head of linked list.
     */
    @tags.LinkedList
    @tags.TwoPointers
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;

        // two runner method
        ListNode fast = dummy, slow = dummy;

        // move the fast runner to forward by n steps
        while (n-- >= 0) {
            // the length of the list is less than n
            if (fast == null) {
                return head;
            }
            fast = fast.next;
        }

        // move fast and slow simultaneously until fast hit the end
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }

        // delete the node next to slow
        slow.next = slow.next.next;

        return dummy.next;
    }

    /**
     * Partition List.
     *
     * Given a linked list and a value x, partition it such that all nodes less
     * than x come before nodes greater than or equal to x.
     *
     * You should preserve the original relative order of the nodes in each of
     * the two partitions.
     *
     * For example, Given 1->4->3->2->5->2 and x = 3, return 1->2->2->4->3->5.
     *
     * @param head:
     *            The first node of linked list.
     * @param x:
     *            an integer
     * @return: a ListNode
     */
    @tags.LinkedList
    @tags.TwoPointers
    public ListNode partition(ListNode head, int x) {
        ListNode dummy1 = new ListNode(0), dummy2 = new ListNode(0);
        ListNode prev1 = dummy1, prev2 = dummy2;

        while (head != null) {
            if (head.val < x) {
                prev1.next = head;
                prev1 = head;
            } else {
                prev2.next = head;
                prev2 = head;
            }
            head = head.next;
        }

        prev1.next = dummy2.next;
        prev2.next = null;

        return dummy1.next;
    }

    /**
     * Remove Duplicates from Sorted List
     *
     * Given a sorted linked list, delete all duplicates such that each element
     * appear only once.
     *
     * For example: Given 1->1->2, return 1->2. Given 1->1->2->3->3, return
     * 1->2->3.
     *
     * Challenge: (hard) How would you solve this problem if a temporary buffer
     * is not allowed? In this case, you don't need to keep the order of nodes.
     *
     * @param ListNode
     *            head is the head of the linked list
     * @return: ListNode head of linked list
     */
    @tags.LinkedList
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null)
            return head;

        ListNode current = head;
        while (current.next != null) {
            if (current.val == current.next.val) {
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
     *
     * @param ListNode
     *            head is the head of the linked list
     * @return: ListNode head of the linked list
     */
    @tags.LinkedList
    public ListNode deleteDuplicatesII(ListNode head) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        head = dummy;

        while (head != null && head.next != null) {
            ListNode current = head.next;
            while (current.next != null && current.next.val == head.next.val) {
                current = current.next;
            }
            if (head.next == current) {
                head = current;
            } else {
                head.next = current.next;
            }
        }

        return dummy.next;
    }

    /**
     * Remove Duplicates from Unsorted List
     *
     * Write code to remove duplicates from an unsorted linked list.
     *
     * Example : Given 1->3->2->1->4. Return 1->3->2->4
     *
     * @param head:
     *            The first node of linked list.
     * @return: head node
     */
    @tags.LinkedList
    public ListNode removeDuplicates(ListNode head) {
        if (head == null) {
            return head;
        }

        ListNode prev = head;
        Set<Integer> set = new HashSet<>();
        set.add(head.val);
        while (prev.next != null) {
            if (!set.contains(prev.next.val)) {
                prev = prev.next;
                set.add(prev.val);
            } else {
                prev.next = prev.next.next;
            }
        }

        return head;
    }

    /**
     * Convert Sorted List to Balanced BST (Convert Sorted List to Binary Search
     * Tree).
     *
     * Given a singly linked list where elements are sorted in ascending order,
     * convert it to a height balanced BST.
     *
     * @param head: The first node of linked list.
     * @return: a tree node
     */
    @tags.LinkedList
    @tags.Recursion
    @tags.DFS
    @tags.Company.Zenefits
    public TreeNode sortedListToBST(ListNode head) {
        ArrayList<TreeNode> treeNodes = new ArrayList<>();
        while (head != null) {
            treeNodes.add(new TreeNode(head.val));
            head = head.next;
        }

        return sortedListToBST(treeNodes, 0, treeNodes.size() - 1);
    }

    private TreeNode sortedListToBST(ArrayList<TreeNode> treeNodes, int start,
            int end) {
        if (start > end) {
            return null;
        }

        int mid = (start + end) >>> 1;
        TreeNode root = treeNodes.get(mid);
        root.left = sortedListToBST(treeNodes, start, mid - 1);
        root.right = sortedListToBST(treeNodes, mid + 1, end);

        return root;
    }

    /**
     * Convert Sorted List to Balanced BST.
     *
     * TODO: Compare this solution.
     */
    @tags.LinkedList
    @tags.Recursion
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
     * Copy List with Random Pointer.
     *
     * A linked list is given such that each node contains an additional random
     * pointer which could point to any node in the list or null. Return a deep
     * copy of the list.
     *
     * Challenge: Could you solve it with O(1) space?
     *
     * @param head:
     *            The head of linked list with a random pointer.
     * @return: A new head of a deep copy of the list.
     */
    @tags.LinkedList
    @tags.HashTable
    @tags.Company.Amazon
    @tags.Company.Bloomberg
    @tags.Company.Microsoft
    @tags.Company.Uber
    public RandomListNode copyRandomList(RandomListNode head) {
        // copy nodes and append copy after each original node
        RandomListNode current = head;
        while (current != null) {
            RandomListNode copy = new RandomListNode(current.label);
            copy.next = current.next;
            current.next = copy;
            current = copy.next;
        }

        // copy random pointer
        current = head;
        while (current != null) {
            if (current.random != null) {
                current.next.random = current.random.next;
            }
            current = current.next.next;
        }

        // separate the 2 lists
        RandomListNode dummy = new RandomListNode(0);
        RandomListNode prev = dummy;
        current = head;
        while (current != null) {
            prev.next = current.next;
            current.next = current.next.next;
            current = current.next;
            prev = prev.next;
        }

        return dummy.next;
    }

    /**
     * Reorder List.
     *
     * Given a singly linked list L: L0¡úL1¡ú¡­¡úLn-1¡úLn, reorder it to:
     * L0¡úLn¡úL1¡úLn-1¡úL2¡úLn-2¡ú¡­
     *
     * You must do this in-place without altering the nodes' values.
     *
     * For example, Given {1,2,3,4}, reorder it to {1,4,2,3}.
     *
     * @param head:
     *            The head of linked list.
     * @return: void
     */
    @tags.LinkedList
    public void reorderList(ListNode head) {
        if (head == null || head.next == null || head.next.next == null) {
            return;
        }

        // find the second half
        ListNode slow = head, fast = head.next;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }

        // break the list, reverse the second half list
        ListNode list = reverse(slow.next);
        slow.next = null;

        // merge the two lists
        while (list != null) {
            ListNode next = list.next;
            list.next = head.next;
            head.next = list;
            head = list.next;
            list = next;
        }
    }

    /**
     * Sort List.
     *
     * Sort a linked list in O(n log n) time using constant space complexity.
     *
     * Example: Given 1->3->2->null, sort it to 1->2->3->null.
     *
     * Challenge: Solve it by merge sort & quick sort separately.
     *
     * Important: Space O(n) solution: copy all nodes to array to avoid cost of
     * findMiddle!!! Collections.sort dumps list elements to an array first.
     *
     * @param head:
     *            The head of linked list.
     * @return: You should return the head of the sorted linked list, using
     *          constant space complexity.
     */
    @tags.LinkedList
    @tags.Sort
    public ListNode sortList(ListNode head) {
        ArrayList<ListNode> list = new ArrayList<>();
        while (head != null) {
            list.add(head);
            head = head.next;
        }

        Collections.sort(list, new Comparator<ListNode>() {
            @Override
            public int compare(ListNode n1, ListNode n2) {
                return n1.val - n2.val;
            }
        });

        list.add(null);
        for (int i = 0; i < list.size() - 1; i++) {
            list.get(i).next = list.get(i + 1);
        }

        return list.get(0);
    }

    /** Sort List - Quick sort */
    @tags.LinkedList
    @tags.Sort
    public ListNode sortListQuick(ListNode head) {
        if (head == null)
            return null;

        // partition
        ListNode less = new ListNode(0);
        ListNode lessTail = less;
        ListNode equal = head;
        ListNode equalTail = equal;
        ListNode greater = new ListNode(0);
        ListNode greaterTail = greater;
        head = head.next;
        while (head != null) {
            if (head.val == equalTail.val) {
                equalTail.next = head;
                equalTail = head;
            } else if (head.val < equalTail.val) {
                lessTail.next = head;
                lessTail = head;
            } else {
                greaterTail.next = head;
                greaterTail = head;
            }
            head = head.next;
        }
        lessTail.next = null;
        equalTail.next = null;
        greaterTail.next = null;

        // sort
        less = sortListQuick(less.next);
        greater = sortListQuick(greater.next);

        // merge
        return merge(less, equal, greater);
    }

    private ListNode merge(ListNode less, ListNode equal, ListNode greater) {
        ListNode dummy = new ListNode(0);
        ListNode prev = dummy;

        // append less
        prev.next = less;
        while (prev.next != null) {
            prev = prev.next;
        }

        // append equal
        prev.next = equal;
        while (prev.next != null) {
            prev = prev.next;
        }

        // append greater
        prev.next = greater;

        return dummy.next;
    }

    /** Sort List - Merge sort */
    @tags.LinkedList
    @tags.Sort
    public ListNode sortListMerge(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        // divide
        ListNode tail = middleNode(head);
        ListNode head2 = tail.next;
        tail.next = null;

        // conquer
        head = sortListMerge(head);
        head2 = sortListMerge(head2);
        return merge(head, head2);
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
     * Linked List Cycle - O(1) space.
     *
     * Given a linked list, determine if it has a cycle in it.
     *
     * Example: Given -21->10->4->5, tail connects to node index 1, return true
     *
     * Challenge: Follow up: Can you solve it without using extra space?
     *
     * @param head:
     *            The first node of linked list.
     * @return: True if it has a cycle, or false
     */
    @tags.TwoPointers
    @tags.LinkedList
    @tags.Company.Amazon
    @tags.Company.Bloomberg
    @tags.Company.Microsoft
    @tags.Company.Yahoo
    public boolean hasCycle(ListNode head) {
        ListNode slow = head, fast = head;

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
     * Space Complexity: O(1)
     *
     * @param head:
     *            The first node of linked list.
     * @return: The node where the cycle begins. if there is no cycle, return
     *          null
     */
    @tags.TwoPointers
    @tags.LinkedList
    public ListNode detectCycle(ListNode head) {
        ListNode slow = head, fast = head;

        // fast double speed to slow
        do {
            if (fast == null || fast.next == null) {
                return null;
            }
            slow = slow.next;
            fast = fast.next.next;
        } while (slow != fast);

        // same pace
        slow = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }

        return fast;
    }

    /**
     * Remove Linked List Elements
     *
     * Remove all elements from a linked list of integers that have value val.
     *
     * Example: Given 1->2->3->3->4->5->3, val = 3, you should return the list
     * as 1->2->4->5
     *
     * @param head
     *            a ListNode
     * @param val
     *            an integer
     * @return a ListNode
     */
    @tags.LinkedList
    public ListNode removeElements(ListNode head, int val) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        head = dummy;
        while (head.next != null) {
            if (head.next.val == val) {
                head.next = head.next.next;
            } else {
                head = head.next;
            }
        }
        return dummy.next;
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
     *
     * @param head
     *            a ListNode
     * @return a ListNode
     */
    @tags.LinkedList
    public ListNode swapPairs(ListNode head) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        head = dummy;
        while (head.next != null && head.next.next != null) {
            ListNode next = head.next;
            head.next = next.next;
            head = head.next;
            next.next = head.next;
            head.next = next;
            head = head.next;
        }
        return dummy.next;
    }

    /**
     * Delete Node in the Middle of Singly Linked List. (Leetcode title: Delete
     * Node in a Linked List.)
     *
     * Example Given 1->2->3->4, and node 3. return 1->2->4
     *
     * @param node:
     *            the node in the list should be deleted
     * @return: nothing
     */
    @tags.LinkedList
    @tags.Source.CrackingTheCodingInterview
    public void deleteNode(ListNode node) {
        if (node == null || node.next == null) {
            throw new NullPointerException("Input node is null or tail.");
        }
        node.val = node.next.val;
        node.next = node.next.next;
    }

    /**
     * Insertion Sort List
     *
     * Sort a linked list using insertion sort.
     *
     * Example Given 1->3->2->0->null, return 0->1->2->3->null.
     *
     * @param head:
     *            The first node of linked list.
     * @return: The head of linked list.
     */
    @tags.LinkedList
    @tags.Sort
    public ListNode insertionSortList(ListNode head) {
        ListNode dummy = new ListNode(0);
        while (head != null) {
            ListNode prev = dummy;
            while (prev.next != null && prev.next.val < head.val) {
                prev = prev.next;
            }

            ListNode insert = head;
            head = head.next;
            insert.next = prev.next;
            prev.next = insert;
        }
        return dummy.next;
    }

    /**
     * Add Two Numbers
     *
     * You are given two linked lists representing two non-negative numbers. The
     * digits are stored in reverse order and each of their nodes contain a
     * single digit. Add the two numbers and return it as a linked list.
     *
     * Example: Given 7->1->6 + 5->9->2. That is, 617 + 295. Return 2->1->9.
     * That is 912. Given 3->1->5 and 5->9->2, return 8->0->8. Input: (2->4->3)
     * + (5->6->4) Output: 7->0->8.
     *
     * @param l1:
     *            the first list
     * @param l2:
     *            the second list
     * @return: the sum list of l1 and l2
     */
    @tags.LinkedList
    @tags.Math
    @tags.HighPrecision
    @tags.Source.CrackingTheCodingInterview
    /** Adding node by node */
    public ListNode addLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode prev = dummy;
        int carry = 0;

        while (carry > 0 || l1 != null || l2 != null) {
            int sum = carry;
            if (l1 != null) {
                sum += l1.val;
                l1 = l1.next;
            }
            if (l2 != null) {
                sum += l2.val;
                l2 = l2.next;
            }
            carry = sum >= 10 ? 1 : 0;
            sum = sum % 10;

            prev.next = new ListNode(sum);
            prev = prev.next;
        }

        return dummy.next;
    }

    /** Add Two Numbers - BigInteger Solution */
    public ListNode addLists2(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode prev = dummy;

        BigInteger n1 = readNumberReverse(l1), n2 = readNumberReverse(l2);
        String n = n1.add(n2).toString();

        for (int i = n.length() - 1; i >= 0; i--) {
            ListNode digit = new ListNode(n.charAt(i) - '0');
            prev.next = digit;
            prev = digit;
        }

        return dummy.next;
    }

    private BigInteger readNumberReverse(ListNode node) {
        BigInteger num = BigInteger.ZERO;
        for (BigInteger time = BigInteger.ONE; node != null;) {
            num = num.add(BigInteger.valueOf(node.val).multiply(time));
            time = time.multiply(BigInteger.TEN);
            node = node.next;
        }
        return num;
    }

    /**
     * Add Two Numbers II
     *
     * You have two numbers represented by a linked list, where each node
     * contains a single digit. The digits are stored in forward order, such
     * that the 1's digit is at the head of the list. Write a function that adds
     * the two numbers and returns the sum as a linked list.
     *
     * Example Given 6->1->7 + 2->9->5. That is, 617 + 295. Return 9->1->2. That
     * is, 912.
     *
     * @param l1:
     *            the first list
     * @param l2:
     *            the second list
     * @return: the sum list of l1 and l2
     */
    @tags.LinkedList
    @tags.Math
    public ListNode addListsII(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode prev = dummy;

        BigInteger n1 = readNumberForward(l1), n2 = readNumberForward(l2);
        String num = n1.add(n2).toString();

        for (int i = 0; i < num.length(); i++) {
            prev.next = new ListNode(num.charAt(i) - '0');
            prev = prev.next;
        }
        return dummy.next;
    }

    private BigInteger readNumberForward(ListNode list) {
        BigInteger num = BigInteger.ZERO;
        while (list != null) {
            BigInteger digit = BigInteger.valueOf(list.val);
            num = num.multiply(BigInteger.TEN).add(digit);
            list = list.next;
        }
        return num;
    }

    /**
     * Merge Two Sorted Lists
     *
     * Merge two sorted (ascending) linked lists and return it as a new sorted
     * list. The new sorted list should be made by splicing together the nodes
     * of the two lists and sorted in ascending order.
     *
     * Example: Given 1->3->8->11->15->null, 2->null , return
     * 1->2->3->8->11->15->null.
     *
     * @param ListNode
     *            l1 is the head of the linked list
     * @param ListNode
     *            l2 is the head of the linked list
     * @return: ListNode head of linked list
     */
    @tags.LinkedList
    @tags.Company.Amazon
    @tags.Company.Apple
    @tags.Company.LinkedIn
    @tags.Company.Microsoft
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode prev = dummy;

        // merge
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                prev.next = l1;
                l1 = l1.next;
            } else {
                prev.next = l2;
                l2 = l2.next;
            }
            prev = prev.next;
        }

        // append the remaining list
        if (l1 != null) {
            prev.next = l1;
        } else if (l2 != null) {
            prev.next = l2;
        }

        return dummy.next;
    }

    /**
     * Swap Two Nodes in Linked List
     *
     * Given a linked list and two values v1 and v2. Swap the two nodes in the
     * linked list with values v1 and v2. It's guaranteed there is no duplicate
     * values in the linked list. If v1 or v2 does not exist in the given linked
     * list, do nothing.
     *
     * Notice: You should swap the two nodes with values v1 and v2. Do not
     * directly swap the values of the two nodes.
     *
     * Example: Given 1->2->3->4->null and v1 = 2, v2 = 4. Return
     * 1->4->3->2->null.
     *
     * @param head
     *            a ListNode
     * @oaram v1 an integer
     * @param v2
     *            an integer
     * @return a new head of singly-linked list
     */
    @tags.LinkedList
    public ListNode swapNodes(ListNode head, int v1, int v2) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        head = dummy;

        ListNode prev1 = null;
        ListNode prev2 = null;
        while (head.next != null && (prev1 == null || prev2 == null)) {
            if (prev1 == null && head.next.val == v1) {
                prev1 = head;
            } else if (prev2 == null && head.next.val == v2) {
                prev2 = head;
            }
            head = head.next;
        }

        if (prev1 != null && prev2 != null) {
            ListNode nv1 = prev1.next;
            ListNode nv2 = prev2.next;
            ListNode after1 = nv1.next;
            ListNode after2 = nv2.next;
            if (nv1 == after2) {
                prev2.next = nv1;
                nv2.next = after1;
                nv1.next = nv2;
            } else if (nv2 == after1) {
                prev1.next = nv2;
                nv1.next = after2;
                nv2.next = nv1;
            } else {
                nv1.next = after2;
                nv2.next = after1;
                prev1.next = nv2;
                prev2.next = nv1;
            }
        }

        return dummy.next;
    }

    /**
     * Palindrome Linked List
     *
     *Implement a function to check if a linked list is a palindrome.
     *
     *Example: Given 1->2->1, return true
     *
     * @param head a ListNode
     * @return a boolean
     */
    @tags.LinkedList
    @tags.TwoPointers
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) {
            return true;
        }

        // split the list at the middle and reverse the second half
        ListNode left = head;
        ListNode middle = middleNode(head);
        ListNode right = middle.next;
        middle.next = null;
        right = reverse(right);

        // check palindrome match
        ListNode lNode = left;
        ListNode rNode = right;
        while (lNode != null && rNode != null) {
            if (lNode.val != rNode.val) {
                return false;
            }
            lNode = lNode.next;
            rNode = rNode.next;
        }

        // restore the list
        right = reverse(right);
        middle.next = right;

        return true;
    }

    /**
     * Convert Binary Search Tree to Doubly Linked List.
     *
     * Convert a binary search tree to doubly linked list with in-order
     * traversal.
     *
     * Example: Given a binary search tree:
     *        4
     *       / \
     *      2   5
     *     / \
     *    1   3
     * return 1<->2<->3<->4<->5.
     *
     * @param root: The root of tree
     * @return: the head of doubly list node
     */
    @tags.LinkedList
    public DoublyListNode bstToDoublyList(TreeNode root) {
        if (root == null) {
            return null;
        }

        DoublyListNode left = bstToDoublyList(root.left);
        DoublyListNode middle = new DoublyListNode(root.val);
        DoublyListNode right = bstToDoublyList(root.right);

        // connect the left
        DoublyListNode head = left;
        if (left != null) {
            while (left.next != null) {
                left = left.next;
            }
            left.next = middle;
            middle.prev = left;
        } else {
            head = middle;
        }

        // connect the right
        if (right != null) {
            middle.next = right;
            right.prev = middle;
        }

        return head;
    }

    /** Iterative solution */
    @tags.LinkedList
    public DoublyListNode bstToDoublyListIter(TreeNode root) {
        if (root == null) {
            return null;
        }

        DoublyListNode dummy = new DoublyListNode(0);
        DoublyListNode prev = dummy;
        Stack<TreeNode> stack = new Stack<>();
        TreeNode current = root;
        while (current != null || !stack.isEmpty()) {
            while (current != null) {
                stack.push(current);
                current = current.left;
            }

            TreeNode node = stack.pop();

            // connect the new node
            prev.next = new DoublyListNode(node.val);
            prev.next.prev = prev;
            prev = prev.next;

            if (node.right != null) {
                current = node.right;
            }
        }

        dummy.next.prev = null;
        return dummy.next;
    }

    /**
     * Rotate List
     *
     * Given a list, rotate the list to the right by k places, where k is
     * non-negative.
     *
     * For example: Given 1->2->3->4->5->NULL and k = 2, return
     * 4->5->1->2->3->NULL.
     *
     * @param head: the List
     * @param k: rotate to the right k places
     * @return: the list after rotation
     */
    @tags.LinkedList
    @tags.TwoPointers
    @tags.BasicImplementation
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null) {
            return null;
        }

        // get list length
        int len = 0;
        ListNode current = head;
        while (current != null) {
            current = current.next;
            len++;
        }

        // split at the node which will be the new tail after rotation
        int position = len - k % len;
        if (position != len) {
            current = head;
            while (--position > 0) {
                current = current.next;
            }

            ListNode rotate = current.next;
            current.next = null;

            current = rotate;
            while (current.next != null) {
                current = current.next;
            }
            current.next = head;
            head = rotate;
        }

        return head;
    }

    /**
     * Reverse Nodes in k-Group
     *
     * Given a linked list, reverse the nodes of a linked list k at a time and
     * return its modified list. If the number of nodes is not a multiple of k
     * then left-out nodes in the end should remain as it is. You may not alter
     * the values in the nodes, only nodes itself may be changed. Only constant
     * memory is allowed.
     *
     * Example: Given this linked list: 1->2->3->4->5. For k = 2, you should
     * return: 2->1->4->3->5. For k = 3, you should return: 3->2->1->4->5.
     *
     * @param head
     *            a ListNode
     * @param k
     *            an integer
     * @return a ListNode
     */
    @tags.LinkedList
    @tags.Company.Facebook
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode prev = dummy;

        while (true) {
            // find next head
            int count = k;
            ListNode current = prev;
            while (count-- > 0) {
                if (current.next == null) {
                    return dummy.next;
                }
                current = current.next;
            }
            ListNode newHead = current.next;
            current.next = null;

            // reverse
            prev.next = reverse(prev.next);
            head.next = newHead;
            prev = head;
            head = newHead;
        }
    }

    public static void main(String[] args) {

    }
}