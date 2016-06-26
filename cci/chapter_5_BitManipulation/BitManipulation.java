package chapter_5_BitManipulation;

public class BitManipulation {

	/**
	 * 5.1 Write a method to insert M into N such that M starts at bit j and
	 * ends at bit i.
	 * 
	 * For example: Input: N = 10000000000, M = 10011, i = 2, j = 6. Output: N =
	 * 10001001100
	 */
	public static int updateBits(int n, int m, int i, int j) {
		int mask = ((1 << 31) >> (30 - j)) & ((1 << 31) >> (31 - i));
		int result = n & mask;
		result |= m << i;
		return result;
	}
	
	public static void testUpdateBits() {
		System.out.println("5.1 testing updateBits()...");
		int n = 1024;
		int m = 19;
		int i = 2;
		int j = 6;
		System.out.println(updateBits(n, m, i, j));
	}

	public static void testAll() {
		testUpdateBits();
	}
	
	public static void main(String[] args) {
		testAll();

	}

}
