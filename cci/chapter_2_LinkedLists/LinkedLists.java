package chapter_2_LinkedLists;

import java.util.Hashtable;
import java.util.Stack;

public class LinkedLists {
	
	/**
	 * 2.1 Removes duplicates from an unsorted linked list
	 * @param head
	 */
	public static void deleteDups(SimpleNode head) {
		Hashtable<Integer, Boolean> ht = new Hashtable<Integer, Boolean>();
		SimpleNode previous = null;
		SimpleNode current = head;
		while (current != null) {
			if (ht.containsKey(current.data)) {
				previous.next = current.next;
			} else {
				ht.put(current.data, true);
				previous = current;
				
			}
			current = current.next;
		} 
	}

	public static void testDeleteDups() {
		System.out.println("2.1 deleteDups()...");
		SimpleNode head = new SimpleNode(0);
		head.add(1);
		head.add(2);
		head.add(2);
		head.add(3);
		head.add(3);
		head.add(2);
		head.add(3);
		head.add(4);
		head.add(7);
		head.add(5);
		head.add(6);
		head.add(7);
		head.add(8);
		SimpleNode.printList(head);
		deleteDups(head);
		SimpleNode.printList(head);
		System.out.println();
	}

	
	/**
	 * 2.2 find the nth to last element of a singly linked list
	 * @param head
	 * @param k
	 */
	public static int nthToLast(SimpleNode head, int k) {
		SimpleNode current = head;
		SimpleNode runner = head;
		for (int i = 0; i <= k; i++) {
			if (runner == null) {
				throw new IllegalArgumentException("k >= LinkedList size.");
			}
			runner = runner.next;
		}
		while (runner != null) {
			runner = runner.next;
			current = current.next;
		}
		return current.data;
	}
	
	public static void testNthToLast() {
		System.out.println("2.2 nthToLast()...");
		SimpleNode head = new SimpleNode(0);
		head.add(1);
		head.add(2);
		head.add(3);
		head.add(4);
		head.add(5);
		head.add(6);
		head.add(7);
		head.add(8);
		SimpleNode.printList(head);
		System.out.println(nthToLast(head, 3));
		System.out.println(nthToLast(head, 2));
		System.out.println(nthToLast(head, 1));
		System.out.println(nthToLast(head, 0));
		System.out.println();
	}

	
	/**
	 * 2.3 delete node in singly LinkedList given only access to that node.
	 * @return false if failed, like the node is in the end; true on success.
	 */
	public static boolean deleteNode(SimpleNode n) {
		if (n == null || n.next == null) {
			return false;
		} else {
			n.data = n.next.data;
			n.next = n.next.next;
			return true;
		}
	}
	
	public static void testDeleteNode() {
		System.out.println("2.3 deleteNode()...");
		SimpleNode head = new SimpleNode(0);
		head.add(1);
		head.add(2);
		head.add(3);
		head.add(4);
		head.add(5);
		head.add(6);
		head.add(7);
		head.add(8);
		SimpleNode.printList(head);
		System.out.println("delete 3: " + deleteNode(head.next.next.next));
		SimpleNode.printList(head);
		System.out.println("delete 4: " + deleteNode(head.next.next.next));
		SimpleNode.printList(head);
		System.out.println();
	}
	

	/**
	 * 2.4 partition a linked list around a value x
	 * Nodes less than x come before all nodes greater or equal to x
	 * @param n
	 * @param x
	 */
	public static SimpleNode partition(SimpleNode head, int x) {
		if (head == null) {
			return head;
		}
		SimpleNode current = head.next;
		SimpleNode tail = head;
		while (current != null) {
			SimpleNode next = current.next;
			if (current.data < x) {
				current.next = head;
				head = current;
				current = next;
				tail.next = current;
			} else {
				current = next;
				tail = tail.next;
			}
		}
		return head;
	}
	
	public static void testPartition() {
		System.out.println("2.4 partition()...");
		SimpleNode head = new SimpleNode(0);
		head.add(7);
		head.add(4);
		head.add(1);
		head.add(8);
		head.add(6);
		head.add(5);
		head.add(3);
		head.add(2);
		SimpleNode.printList(head);
		head = partition(head, 4);
		SimpleNode.printList(head);
		System.out.println();
	}


	/**
	 * 2.5 adds the two numbers and returns the sum as a linked list
	 * @param n1
	 * @param n2
	 */
	public static SimpleNode addLists(SimpleNode n1, SimpleNode n2, int carry) {
		// We are done if both list is 0 and carry is 0
		if (n1 == null && n2 == null && carry == 0) {
			return null;
		}
		
		// smart optimization
		int sum = carry;
		if (n1 != null) {
			sum += n1.data;
		}
		if (n2 != null) {
			sum += n2.data;
		}

		// recurse
		int digit = sum % 10;
		carry = sum / 10;
		SimpleNode n3 = new SimpleNode(digit);
		n3.next = addLists(n1 == null ? null : n1.next, // ÕæÔÞ
						   n2 == null ? null : n2.next, carry);
		return n3;
	}
	
	public static void testAddLists() {
		System.out.println("2.5 addLists()...");
		SimpleNode n1 = new SimpleNode(3);
		n1.add(1);
		n1.add(5);
		SimpleNode n2 = new SimpleNode(5);
		n2.add(9);
		n2.add(4);
		SimpleNode.printList(n1);
		System.out.println("+");
		SimpleNode.printList(n2);
		System.out.println("=");
		SimpleNode.printList(addLists(n1, n2, 0));
		System.out.println();
	}


	/**
	 * 2.6 Given a circular linked list, returns the beginning node in the loop
	 * @param n
	 * @return
	 */
	public static SimpleNode findBeginning(SimpleNode head) {
		SimpleNode slow = head;
		SimpleNode fast = head;
		
		/* Find meeting point */
		while(fast != null && fast.next != null) {
			slow = slow.next;
			fast = fast.next.next;
			/* this collision cannot be put in while condition */
			if (slow == fast) { // this collision cannot be put in the 
				break;
			}
		}
		
		/* Error check, no loop */
		if (slow != fast) {
			return null;
		}
		
		/* Move slow to Head, keep fast at Meeting Point. */
		/* Each are k steps from the Loop Start. */
		slow = head;
		while (slow != fast) {
			slow = slow.next;
			fast = fast.next;
		}
		
		return slow;
	}
	
	public static void testFindBeginning() {
		System.out.println("2.6 findBeginning()...");
		SimpleNode head = new SimpleNode(0);
		head.add(0);
		head.add(0);
		head.add(0);
		head.add(0);
		head.add(0);
		head.add(0);
		head.add(0);
		head.add(0);
		SimpleNode begin = new SimpleNode(1);
		head.addNode(begin);
		begin.add(2);
		begin.add(3);
		begin.add(4);
		begin.add(5);
		begin.add(6);
		begin.add(7);
		begin.addNode(begin);
		System.out.println(findBeginning(head).data);
		System.out.println();
	}
	
	
	/**
	 * 2.7 check if a linked list is palindrome
	 * @param head
	 * @return
	 */
	public static boolean isPalindrome(SimpleNode head) {
		SimpleNode slow = head;
		SimpleNode fast = head;
		Stack<Integer> stack = new Stack<Integer>();
		
		while (fast != null && fast.next != null) {
			stack.push(slow.data);
			slow = slow.next;
			fast = fast.next.next;
		}
		
		if (fast != null) {
			slow = slow.next;
		}
		
		fast = head;
		while (slow != null) {
			if (slow.data != stack.pop().intValue()) {
				return false;
			}
			slow = slow.next;
		}
		return true;
	}

	public static void testIsPalindrome() {
		System.out.println("2.7 isPalindrome()...");
		SimpleNode head1 = new SimpleNode(0);
		head1.add(1);
		head1.add(2);
		head1.add(3);
		head1.add(2);
		head1.add(1);
		head1.add(0);
		SimpleNode.printList(head1);
		System.out.println(isPalindrome(head1));
		
		SimpleNode head2 = new SimpleNode(0);
		head2.add(1);
		head2.add(2);
		head2.add(3);
		head2.add(4);
		head2.add(3);
		head2.add(2);
		head2.add(1);
		head2.add(0);
		SimpleNode.printList(head2);
		System.out.println(isPalindrome(head2));
		
		SimpleNode head3 = new SimpleNode(0);
		head3.add(1);
		head3.add(2);
		head3.add(3);
		head3.add(2);
		head3.add(1);
		head3.add(1);
		SimpleNode.printList(head3);
		System.out.println(isPalindrome(head3));
		
		SimpleNode head4 = new SimpleNode(0);
		head4.add(1);
		head4.add(2);
		head4.add(3);
		head4.add(4);
		head4.add(3);
		head4.add(2);
		head4.add(1);
		head4.add(1);
		SimpleNode.printList(head4);
		System.out.println(isPalindrome(head4));
		
		System.out.println();
	}
	
	
	public static void testAll() {
		testDeleteDups();
		testNthToLast();
		testDeleteNode();
		testPartition();
		testAddLists();
		testFindBeginning();
		testIsPalindrome();
	}
	
	public static void main(String[] args) {
		testAll();
	}
}
