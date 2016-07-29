package categories;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;

public class LinkedLists {

    // ******************************** MODELS ********************************

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

    /** Definition for undirected graph. */
    class UndirectedGraphNode {
        int label;
        ArrayList<UndirectedGraphNode> neighbors;

        UndirectedGraphNode(int x) {
            label = x;
            neighbors = new ArrayList<UndirectedGraphNode>();
        }
    }

    // ******************************* TEMPLATE *******************************

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
    @tags.Company.Facebook
    @tags.Company.Uber
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
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode newHead = reverseList(head.next);
        head.next.next = head;
        head.next = null;

        return newHead;
    }

    // ******************************* PROBLEMS *******************************

    /**
     * Remove Nth Node From End of List
     *
     * Given a linked list, remove the nth node from the end of list and return
     * its head.
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
        ListNode fast = head, slow = dummy;

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
        ListNode list1 = new ListNode(-1);
        ListNode tail1 = list1;
        ListNode list2 = new ListNode(-1);
        ListNode tail2 = list2;

        while (head != null) {
            if (head.val < x) {
                tail1.next = head;
                tail1 = head;
            } else {
                tail2.next = head;
                tail2 = head;
            }
            head = head.next;
        }

        tail1.next = list2.next;
        tail2.next = null;
        return list1.next;
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
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        head = dummy;

        while (head != null && head.next != null) {
            ListNode lastDup = head.next;
            while (lastDup.next != null && lastDup.val == lastDup.next.val) {
                lastDup = lastDup.next;
            }
            if (head.next != lastDup) {
                head.next = lastDup.next;
            } else {
                head = head.next;
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
     * Convert Sorted List to Balanced BST
     *
     * Given a singly linked list where elements are sorted in ascending order,
     * convert it to a height balanced BST.
     */
    @tags.LinkedList
    @tags.Recursion
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
     * Convert Sorted List to Balanced BST
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
     * Copy List with Random Pointer
     *
     * A linked list is given such that each node contains an additional random
     * pointer which could point to any node in the list or null. Return a deep
     * copy of the list.
     *
     * @param head:
     *            The head of linked list with a random pointer.
     * @return: A new head of a deep copy of the list.
     */
    @tags.LinkedList
    @tags.HashTable
    @tags.Company.Uber
    public RandomListNode copyRandomList(RandomListNode head) {
        RandomListNode dummy = new RandomListNode(0);
        dummy.next = head;

        // copy nodes and append copy after each original node
        while (head != null) {
            RandomListNode next = head.next;
            head.next = new RandomListNode(head.label);
            head.next.next = next;
            head = next;
        }

        // copy random pointer
        head = dummy.next;
        while (head != null) {
            if (head.random != null) {
                head.next.random = head.random.next;
            }
            head = head.next.next;
        }

        // separate the 2 lists
        RandomListNode pre = dummy;
        head = dummy.next;
        while (head != null) {
            pre.next = head.next;
            pre = pre.next;
            head.next = pre.next;
            head = head.next;
        }
        pre.next = null;

        return dummy.next;
    }

    /**
     * Merge k Sorted Lists
     *
     * Merge k sorted linked lists and return it as one sorted list. Analyze and
     * describe its complexity.
     *
     * @param lists:
     *            a list of ListNode
     * @return: The head of one sorted list.
     */
    @tags.DivideAndConquer
    @tags.LinkedList
    @tags.PriorityQueue
    @tags.Heap
    @tags.Company.Airbnb
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Company.LinkedIn
    @tags.Company.Twitter
    @tags.Company.Uber
    public ListNode mergeKLists(List<ListNode> lists) {
        if (lists == null || lists.size() == 0) {
            return null;
        }

        // construct min heap
        PriorityQueue<ListNode> pq = new PriorityQueue<ListNode>(lists.size(),
                new Comparator<ListNode>() {
                    @Override
                    public int compare(ListNode n1, ListNode n2) {
                        return n1.val - n2.val;
                    }
                });

        // insert head node of each list
        for (ListNode list : lists) {
            if (list != null) {
                pq.offer(list);
            }
        }

        // merge
        ListNode dummy = new ListNode(0);
        ListNode prev = dummy;
        while (!pq.isEmpty()) {
            prev.next = pq.poll();
            prev = prev.next;
            if (prev.next != null) {
                pq.offer(prev.next);
            }
        }

        return dummy.next;
    }

    /**
     * Reorder List
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
        list = reverse(list);

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

    /**
     * Sort List
     *
     * Sort a linked list in O(n log n) time using constant space complexity.
     *
     * Important: Space O(n) solution: copy all nodes to array to avoid cost of
     * findMiddle!!! Collections.sort dumps list elements to an array first.
     */
    @tags.LinkedList
    /** Quick sort */
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

    /**
     * Sort List - Merge sort
     */
    public ListNode sortListMerge(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        // divide
        ListNode tail = findMiddle(head);
        ListNode head2 = tail.next;
        tail.next = null;

        // conquer
        head = sortListMerge(head);
        head2 = sortListMerge(head2);
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
     * Linked List Cycle
     *
     * Given a linked list, determine if it has a cycle in it.
     *
     * Space Complexity: O(1)
     *
     * @param head:
     *            The first node of linked list.
     * @return: True if it has a cycle, or false
     */
    @tags.TwoPointers
    @tags.LinkedList
    public boolean hasCycle(ListNode head) {
        ListNode slow = head, fast = head;
        do {
            if (fast == null || fast.next == null) {
                return false;
            }
            slow = slow.next;
            fast = fast.next.next;
        } while (slow != fast);
        return true;
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
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode fast = dummy, slow = dummy;

        // fast double speed to slow
        do {
            if (fast == null || fast.next == null) {
                return null;
            }
            slow = slow.next;
            fast = fast.next.next;
        } while (fast != slow);

        // same pace
        slow = dummy;
        while (fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }

        return fast;
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
     * Given 7->1->6 + 5->9->2. That is, 617 + 295. Return 2->1->9. That is 912.
     * Given 3->1->5 and 5->9->2, return 8->0->8. Input: (2 -> 4 -> 3) + (5 -> 6
     * -> 4) Output: 7 -> 0 -> 8.
     *
     * @param l1:
     *            the first list
     * @param l2:
     *            the second list
     * @return: the sum list of l1 and l2
     */
    @tags.LinkedList
    @tags.Math
    @tags.Source.CrackingTheCodingInterview
    /** Adding node by node */
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
            if (carry != 0)
                current.next = new ListNode(carry);
            return l.next;
        } else if (l1 != null) { // l1 still need to be added
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
        if (carry != 0)
            current.next = new ListNode(carry);
        return l.next;
    }

    /** Add Two Numbers - BigInteger Solution */
    public ListNode addLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode prev = dummy;

        BigInteger n1 = readNumber(l1), n2 = readNumber(l2);
        String n = n1.add(n2).toString();

        for (int i = n.length() - 1; i >= 0; i--) {
            ListNode digit = new ListNode(n.charAt(i) - '0');
            prev.next = digit;
            prev = digit;
        }

        return dummy.next;
    }

    private BigInteger readNumber(ListNode node) {
        BigInteger num = BigInteger.ZERO;
        for (BigInteger time = BigInteger.valueOf(1); node != null;) {
            num = num.add(BigInteger.valueOf(node.val).multiply(time));
            time = time.multiply(BigInteger.TEN);
            node = node.next;
        }
        return num;
    }

    /**
     * Nth to Last Node in List
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
    @tags.Company.LinkedIn
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
        }
        if (l2 != null) {
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

        ListNode pre1 = null;
        ListNode pre2 = null;
        while (head.next != null && (pre1 == null || pre2 == null)) {
            if (pre1 == null && head.next.val == v1) {
                pre1 = head;
            } else if (pre2 == null && head.next.val == v2) {
                pre2 = head;
            }
            head = head.next;
        }

        if (pre1 != null && pre2 != null) {
            ListNode nv1 = pre1.next;
            ListNode nv2 = pre2.next;
            ListNode after1 = nv1.next;
            ListNode after2 = nv2.next;
            if (nv1 == after2) {
                pre2.next = nv1;
                nv2.next = after1;
                nv1.next = nv2;
            } else if (nv2 == after1) {
                pre1.next = nv2;
                nv1.next = after2;
                nv2.next = nv1;
            } else {
                nv1.next = after2;
                nv2.next = after1;
                pre1.next = nv2;
                pre2.next = nv1;
            }
        }

        return dummy.next;
    }

    // ------------------------------ OLD ------------------------------------

    /**
     * 3. Reverse Nodes in k-Group
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
     * 13. Rotate List
     *
     * Given a list, rotate the list to the right by k places, where k is
     * non-negative.
     *
     * For example: Given 1->2->3->4->5->NULL and k = 2, return
     * 4->5->1->2->3->NULL.
     */
    public ListNode rotateRight(ListNode head, int n) {
        if (head == null)
            return null;

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
        if (fast == head)
            return head;

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
     * Clone Graph
     *
     * If use iterative solution, copy node first, then copy connection use map
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

    public static void main(String[] args) {

    }
}