/**
 * Author: Guangcheng Lu
 * Andrew ID: guangchl
 * 
 * @ ALL RIGHT RESERVED
 */

package data_structure;

import java.util.Arrays;
import java.util.Comparator;
import java.util.PriorityQueue;

/**
 * -----------------------------------------------------------
 * Given an input sequence of n integers, each of which has a value between
 * 1 and 100 (inclusive), write a function that outputs the sum of the k
 * largest integers. Use a language of your choice for the implementation,
 * but ideally C++ or Java.
 * 
 * Additional questions (answered in English, no coding needed)
 * 
 *    * What is the complexity of your algorithm? Do you think it is the best?
 *    * What if each value is between 1 and m, where m is a variable?
 * 
 * If there are unspecific elements in this coding question (e.g. data
 * structures), choose whatever that is reasonable to you.
 * -------------------------------------------------------------------
 */
public class MaxMinHeap {

	/** Solution 1 - MaxHeap */
	public int kSumMaxHeap(int[] A, int k) {
		PriorityQueue<Integer> pq = new PriorityQueue<Integer>(A.length,
				new Comparator<Integer>() {
					@Override
					public int compare(Integer o1, Integer o2) {
						if (o1 > o2) {
							return -1;
						} else if (o1 < o2) {
							return 1;
						}

						return 0;
					}
				});

		// construct the MaxHeap
		for (Integer i : A) {
			pq.add(i);
		}

		// aggregation
		int sum = 0;
		for (int i = 0; i < k; i++) {
			sum += pq.remove();
		}

		return sum;
	}
	
	/** Solution 2 - MinHeap */
	public int kSumMinHeap(int[] A, int k) {
		PriorityQueue<Integer> pq = new PriorityQueue<Integer>(k);

		// construct the MinHeap
		for (int i = 0; i < k; i++) {
			pq.add(A[i]);
		}

		// update the MinHeap
		for (int i = k; i < A.length; i++) {
			if (pq.peek() < A[i]) {
				pq.remove();
				pq.add(A[i]);
			}
		}
		
		// aggregation
		int sum = 0;
		for (Integer i : pq) {
			sum += i;
		}

		return sum;
	}

	/** test case */
	public void test() {
		int[] A = { 1, 4, 2, 7, 5, 8, 9, 3, 9 };
		int[] ks = { 1, 2, 3, 4, 5};
		System.out.println(Arrays.toString(A) + "\n");
		
		for (Integer k : ks) {
			System.out.println("k = " + k);
			System.out.println("MaxHeap: " + kSumMaxHeap(A, k));
			System.out.println("MinHeap: " + kSumMinHeap(A, k));
			System.out.println();
		}
	}

	public static void main(String[] args) {
		MaxMinHeap heap = new MaxMinHeap();
		heap.test();
	}
}

/**
 * Follow up discussion:
 * 
 * 1. My solutions:
 *   1) MaxHeap:
 *      a) Build a MaxHeap in O(n)
 *      b) Extract max element k times elements from the MaxHeap O(klogn)
 *   
 *      Time complexity: O(n + klogn)
 *      Space complexity: O(n)
 *   
 *   2) MinHeap
 *      a) Build a MinHeap in O(k)
 *      b) For every of remaining n - k elements, insert it to the heap if it is 
 *         larger than the minimum in the heap, at the same time delete the 
 *         minimum to maintain the heap size, which is O((n - k) * logk)
 *      c) add k integer in the heap to get result in O(k)
 *      
 *      Time complexity: O(k + (n - k) * logk)
 *      Space complexity: O(k)
 * 
 * 
 * 2. Is my solution the best?
 *   Probably not, since algorithms can always be improved.
 *   
 *   However, we may have different choice depends on different situations and 
 *   requirements.
 * 
 * 
 * 3. I list several alternative solutions below.
 *   1) Bucket Sort
 *      Actually this Method can gain the better time complexity as O(n).  
 *      For space complexity, it is O(100) as we need 100 buckets for 1 to 100.
 *      
 *      However, the trade off is space complexity and this method will depends 
 *      on the number 100. In the following 2 situations, we will not want this 
 *      solution:
 *      
 *      a) we want to have a algorithm can be more widely used
 *      b) we do care about the space, and n is less than 100 (in this problem)
 *      
 *      But of course, if we have lot space available or we don't care about 
 *      space or n is far more larger than 100, Bucket sort is a good choice.
 *      
 *   2) Quick Sort Partition
 *      a) Choose a integer in the array to do the partition, which is O(n).
 *      b) Since we partition the array into 2 parts around pivot value, we only
 *         have 1 part left to handle.
 *      c) loop to do the same thing for the left part until we find k numbers
 *      
 *      Time complexity on average: O(n + n/2 + n/4 + ... + 1), which is O(n)
 *      Space complexity: O(1)
 *   
 *      The trade off of this solution is the same as quick sort, which is we 
 *      have a worst case. We may choose bad pivot every time and hurt the time
 *      complexity as O(n + n-1 + n-2 + ... + n-k+1), which is O((2n-k+1) * k).
 *      
 *      However, in the real world test, quick sort is pretty good, usually 
 *      faster than other O(nlogn)
 */
