package categories;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.PriorityQueue;

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
     */
    @tags.LinkedList
    @tags.Site.LeetCode
    @tags.Site.LintCode
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
    @tags.Site.LeetCode
    @tags.Site.LintCode
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

    // ------------------------------ OLD ------------------------------------

    /**
     * 1. Reverse Linked List II
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
        if (head == null || m >= n || m < 0) {
            return head;
        }

        ListNode dummy = new ListNode(0);
        dummy.next = head;
        head = dummy;

        // find the node before reverse part
        n -= m;
        while (m-- > 1 && head != null) {
            head = head.next;
        }

        // list is shorter than or equals to m
        if (head == null || head.next == null) {
            return dummy.next;
        }

        // reverse
        ListNode tail = head.next;
        ListNode prev = tail;
        ListNode current = prev.next;
        while (current != null && n-- > 0) {
            ListNode next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }

        // reconnect
        head.next = prev;
        tail.next = current;

        return dummy.next;
    }

    /**
     * 2. Swap Nodes in Pairs
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
     * 4. Linked List Cycle
     *
     * Given a linked list, determine if it has a cycle in it.
     *
     * Space Complexity: O(1)
     */
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }

        // set two runners
        ListNode slow = head;
        ListNode fast = head;

        // fast runner move 2 steps at one time while slow runner move 1 step,
        // if traverse to a null, there must be no loop, use do while to make
        // sure we can move at least once
        do {
            slow = slow.next;
            fast = fast.next.next;
        } while (fast != null && fast.next != null && slow != fast);

        return fast == slow;
    }

    /**
     * 5. Linked List Cycle II
     *
     * Given a linked list, return the node where the cycle begins. If there is
     * no cycle, return null.
     *
     * Space Complexity: O(1)
     */
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) {
            return null;
        }

        // set two runners
        ListNode slow = head;
        ListNode fast = head;

        // first time meet: fast runner move 2 steps at one time while slow
        // runner move 1 step,
        do {
            slow = slow.next;
            fast = fast.next.next;
        } while (fast != null && fast.next != null && slow != fast);

        // if stopped by null, indicating no loop
        if (slow != fast) {
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
     * 7. Merge Two Sorted Lists
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
     * 8. Merge k Sorted Lists
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

        // merge
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
     * 12. Reorder List
     * 
     * Given a singly linked list L: L0¡úL1¡ú¡­¡úLn-1¡úLn, reorder it to:
     * L0¡úLn¡úL1¡úLn-1¡úL2¡úLn-2¡ú¡­
     * 
     * You must do this in-place without altering the nodes' values.
     * 
     * For example, Given {1,2,3,4}, reorder it to {1,4,2,3}.
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
     * 14. Insertion Sort List
     * 
     * Sort a linked list using insertion sort.
     */
    public ListNode insertionSortList(ListNode head) {
        if (head == null)
            return null;

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
     * 15. Sort List
     * 
     * Sort a linked list in O(n log n) time using constant space complexity.
     * 
     * Important: Space O(n) solution: copy all nodes to array to avoid cost of
     * findMiddle!!!
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
     * 16. Copy List with Random Pointer
     */
    public RandomListNode copyRandomList(RandomListNode head) {
        if (head == null) {
            return null;
        }

        // copy the main list and append each copy to its original node
        // list length is doubled
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
            current.random = (current.random == null) ? null
                    : current.random.next;
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