package pastinterviews;

import java.util.ArrayList;

public class PureStorage {
	
	public class ListNode {
        int val;
        ListNode next;
    
        public ListNode(int x) {
            val = x;
            next = null;
        }
    }

	/**
	 * Remove all elements from a linked list of integers that have value N
	 */
	public ListNode removeNode(int val, ListNode head) {
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
	 * find all two elements pairs which have difference D in a sorted array.
	 */
	public ArrayList<ArrayList<Integer>> findPairs(int[] A, int D) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (A == null || A.length < 2) {
			return result;
		}
		
		int first = 0;
		int second = 1;
		while (second < A.length) {
			if (A[second] - A[first] < D || first == second) {
				second++;
			} else if (A[second] - A[first] > D) {
				first++;
			} else {
				ArrayList<Integer> pair = new ArrayList<Integer>();
				pair.add(A[first]);
				pair.add(A[second]);
				result.add(pair);
				second++;
			}
		}
		
		return result;
	}
	
	public void test() {
		int[] A = {0, 1, 3, 4, 4, 5, 7, 8, 9, 12, 15};
		int D = 0;
		for (ArrayList<Integer> pair : findPairs(A, D)) {
			System.out.println(pair);
		}
	}
	
	public static void main(String[] args) {
		PureStorage sol = new PureStorage();
		sol.test();
	}

}
