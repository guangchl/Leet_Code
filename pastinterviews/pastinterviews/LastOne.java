package pastinterviews;

/**
 * Problem from POPSUGAR
 * 
 * Take a second to imagine that you are in a room with 100 chairs arranged in a
 * circle. These chairs are numbered sequentially from One to One Hundred. At
 * some point in time, the person in chair #1 will be told to leave the room.
 * The person in chair #2 will be skipped, and the person in chair #3 will be
 * told to leave. Next to go is person in chair #6. In other words, 1 person
 * will be skipped initially, and then 2 people, 3 people, 4 people.. and so on.
 * This pattern of skipping will keep going around the circle until there is
 * only one person remaining.. the survivor. Write a program to figure out which
 * chair the survivor is sitting in.
 * 
 * Author: Guangcheng Lu
 * All rights reserved.
 */

public class LastOne {
	public class ListNode {
        int val;
        ListNode next;
    
        public ListNode(int x) {
            val = x;
            next = null;
        }
    }
	
	public int test(int n) {
		if (n <= 0) {
			System.err.println("Input should be a positive integer.");
			return -1;
		} else if (n <= 2) {
			return n;
		}
		
		// construct a circle linked list
		System.out.println("Constructing the circle linked list...\n");
		ListNode head = new ListNode(1);
		ListNode temp = head;
		for (int i = 2; i <= n; i++) {
			temp.next = new ListNode(i);
			temp = temp.next;
		}
		temp.next = head;
		head = temp;
		
		// how many chairs should be skipped to remove next one
		int size = n;
		int skip = 0;
		
		// add timer around the algorithm
		System.out.println("Calculating the survivor...");
		long start = System.currentTimeMillis(); // Timer starts
		
		// every outer loop remove one list node
		while (head != head.next) {
			int i = skip;
			skip++;
			i = i % size; // mod to reduce the skip cost
			while (i-- > 0) {
				head = head.next;
			}
			
//			System.out.println(head.next.val + " is deleted. size: " + size
//					+ " skip: " + (skip - 1)); // print to show detail
			head.next = head.next.next;
			size--;
		}
		
		System.out.println("Time elapse: " + (System.currentTimeMillis() - start) + " milliseconds"); // Timer ends
		return head.val;
	}
	
	public static void main(String[] args) {
		int n = 100;
		LastOne lastOne = new LastOne();
		
		System.out.println("Final survivor -> " + lastOne.test(n));
	}
}
