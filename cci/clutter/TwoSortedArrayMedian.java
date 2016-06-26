package clutter;

public class TwoSortedArrayMedian {
	private static int middleIndex(int left, int right) {
		return (left + right) / 2;
	}
	
	public static double findMedianUtil(int[] a, int aLeft, int aRight, int[] b, int bLeft, int bRight) {
		int aMiddle = middleIndex(aLeft, aRight);
		int bMiddle = middleIndex(bLeft, bRight);
		if (a[aMiddle] < b[bMiddle]) {
			
		} else if (a[aMiddle] == b[bMiddle]) {
			
		}
		return 0;
	}
	
	public static double findMedian(int[] a, int[] b) {
		if (a.length > b.length) {
			return findMedianUtil(b, 0, b.length - 1, a, 0, a.length - 1);
		} else {
			return findMedianUtil(a, 0, a.length - 1, b, 0, b.length - 1);
		}
	}
	
	public static void main(String[] args) {
		int[] a = {0, 1, 3, 6, 8, 10, 17, 19, 22};
		int[] b = {1, 3, 4, 5, 7, 8, 12, 13, 16, 20, 25};

		for (int i : a) {
			System.out.print(i + " ");
		}
		for (int i : b) {
			System.out.print(i + " ");
		}
		System.out.println("Median: " + findMedian(a, b));
		
		int[] c = {16, 200};

		for (int i : a) {
			System.out.print(i + " ");
		}
		for (int i : c) {
			System.out.print(i + " ");
		}
		System.out.println("Median: " + findMedian(a, b));
	}

}
