package chapter_1_ArraysAndStrings;

import java.util.Formatter;

public class ArraysAndStrings {

	/**
	 * 1.1 is UniqueChars()...
	 * @param s
	 * @return
	 */
	public static boolean isUniqueChars(String s) {
		if (s.length() > 256) {
			return false;
		}

		boolean[] ascii = new boolean[256]; // totally 256 character in ascii
		for (int i = 0; i < s.length(); i++) {
			int val = s.charAt(i);
			if (ascii[val]) {
				return false;
			}
			ascii[val] = true;
		}
		return true;
	}

	public static void testIsUniqueChars() {
		System.out.println("1.1");
		String s1 = "I am Guangcheng.";
		System.out.println(!isUniqueChars(s1));
		String s2 = "I'm Guangchl.";
		System.out.println(isUniqueChars(s2));
		System.out.println();
	}

	
	/**
	 * This problem is 1.3 from 4th edition. Here use the char[] as input
	 * because String is immutable
	 */
	public static void removeDuplicate(char[] s) {
		/* this branch is significant */
		if (s == null) {
			return;
		}

		boolean[] ascii = new boolean[256];
		int tail = 0;

		for (int i = 0; i < s.length; i++) {
			int val = s[i];
			if (ascii[val] == true) {
				continue;
			} else {
				ascii[val] = true;
				s[tail] = s[i];
				tail++;
			}
		}
		/* terminate the string with null */
		if (tail < s.length - 1) {
			s[tail] = 0; // same as s[tail] = '\0';
		}
	}

	public static void testRemoveDuplicate() {
		System.out.println("1.3(4th Edition)");
		char[] s1 = { 'a', 'b', 'c', 'd' };
		removeDuplicate(s1);
		System.out.println(s1);
		char[] s2 = { 'a', 'a', 'a', 'a' };
		removeDuplicate(s2);
		System.out.println(s2);
		char[] s3 = null;
		removeDuplicate(s3);
		// System.out.println(s3);
		char[] s4 = {};
		removeDuplicate(s4);
		System.out.println(s4);
		char[] s5 = { 'a', 'a', 'a', 'b', 'b', 'b' };
		removeDuplicate(s5);
		System.out.println(s5);
		char[] s6 = { 'a', 'b', 'a', 'b', 'a', 'c' };
		removeDuplicate(s6);
		System.out.println(s6);
		System.out.println();
	}

	
	/**
	 * 1.3
	 * @param s1
	 * @param s2
	 * @return
	 */
	public static boolean checkPermutation(String s1, String s2) {
		int l1 = s1.length();
		int l2 = s2.length();
		if (l1 != l2) {
			return false;
		} else {
			int[] ascii = new int[256];

			char[] a1 = s1.toCharArray();
			for (int i = 0; i < l1; i++) {
				ascii[a1[i]]++;
			}

			char[] a2 = s2.toCharArray();
			for (int i = 0; i < l2; i++) {
				if (--ascii[a2[i]] < 0) { // this line is good
					return false;
				}
			}
			return true;
		}
	}

	public static void testCheckPermutation() {
		System.out.println("1.3");
		String s1 = "abcd";
		String s2 = "ddba";
		String s3 = "dbca";
		String s4 = "aaaaaaa";
		System.out.println(!checkPermutation(s1, s2));
		System.out.println(checkPermutation(s1, s3));
		System.out.println(!checkPermutation(s1, s4));
		System.out.println();
	}

	
	/**
	 * 1.4 replace all occurrence of space character to "%20" in a String
	 * 
	 * @param s
	 * @return a new String
	 */
	public static void replaceSpace(char[] str, int length) {
		int spaceCounter = 0;
		for (int i = 0; i < length; i++) {
			if (str[i] == ' ') {
				spaceCounter++;
			}
		}
		int newLength = length + spaceCounter * 2;
		str[newLength] = '\0';
		for (int i = length - 1, j = newLength - 1; i >= 0 && j >= 0;) {
			if (str[i] == ' ') {
				i--;
				str[j--] = '0';
				str[j--] = '2';
				str[j--] = '%';
			} else {
				str[j--] = str[i--];
			}
		}
	}

	public static void testReplaceSpace() {
		System.out.println("1.4");
		char[] s1 = "I am Guangcheng Lu.                       ".toCharArray();
		char[] s2 = "  1 2 3 4 5 6 7                           ".toCharArray();
		replaceSpace(s1, 19);
		replaceSpace(s2, 15);
		System.out.println(s1);
		System.out.println(s2);
		System.out.println();
	}

	
	/**
	 * 1.5 basic string compression using the counts of repeated characters
	 * 
	 * @param s
	 */
	public static String simpleStringCompression(String s) {
		if (s.length() <= 2) {
			return s;
		}
		
		int newLength = 2;
		/* count new length */
		for (int i = 1; i < s.length(); i++) {
			if (s.charAt(i) == s.charAt(i - 1)) {
				continue;
			} else {
				newLength += 2;
			}
		}
		
		StringBuffer sb = new StringBuffer();
		if (newLength < s.length()) {
			int counter = 1;
			int i;
			for (i = 1; i < s.length(); i++) {
				if (s.charAt(i) == s.charAt(i - 1)) {
					counter++;
				} else {
					sb.append(s.charAt(i - 1));
					sb.append(counter);
					counter = 1;
				}
			}
			sb.append(s.charAt(i - 1));
			sb.append(counter);
			return sb.toString();
		} else {
			return s;
		}
	}

	public static void testSimpleStringCompression() {
		System.out.println("1.5");
		String s1 = "aabcccccaaa";
		String s2 = "aabbccddee";

		System.out.println(s1);
		System.out.println(simpleStringCompression(s1));

		System.out.println(s2);
		System.out.println(simpleStringCompression(s2));

		System.out.println();
	}

	
	/**
	 * 1.6 rotate a image Matrix by 90 degree clockwise
	 */
	public static void rotateImageMatrix(int[][] matrix, int n) {
		int temp;
		for (int i = 0; i < (n + 1) / 2; i++) {
			for (int j = i; j < n - i - 1; j++) {
				temp = matrix[i][j];
				matrix[i][j] = matrix[n - j - 1][i];
				matrix[n - j - 1][i] = matrix[n - i - 1][n - j - 1];
				matrix[n - i - 1][n - j - 1] = matrix[j][n - i - 1];
				matrix[j][n - i - 1] = temp;
			}
		}
	}
	
	private static String visualizeMatrix(int[][] matrix) {
		String matrixStr = "";
		Formatter myFormatter = new Formatter();
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[0].length; j++) {
				myFormatter.format("%3d", matrix[i][j]);
			}
			myFormatter.format("\n");
		}
		matrixStr += myFormatter;
		myFormatter.close();
		return matrixStr;
	}
	
	public static void testRotateImageMatrix() {
		System.out.println("1.6 rotateImageMatrix()...");
		int[][] matrix = {{11, 12, 13, 14}, 
						  {21, 22, 23, 24}, 
						  {31, 32, 33, 34}, 
						  {41, 42, 43, 44}};
		System.out.println("Old matrix:\n" + visualizeMatrix(matrix));
		rotateImageMatrix(matrix, matrix[0].length);
		System.out.println("New matrix:\n" + visualizeMatrix(matrix));
		System.out.println();
	}
	
	
	/**
	 * 1.7 if an element in m*n matrix is 0, set its entire row and column to 0
	 */
	public static void setZeros(int[][] matrix) {
		int rowNum = matrix.length, columnNum = matrix[0].length;
		int[] row = new int[rowNum];
		int[] column = new int[columnNum];
		for (int i = 0; i < rowNum; i++) {
			for (int j = 0; j < columnNum; j++) {
				if (matrix[i][j] == 0) {
					row[i] = column[j] = 1;
				}
			}
		}
		for (int i = 0; i < rowNum; i++) {
			if (row[i] == 1) {
				for (int j = 0; j < columnNum; j++) {
					matrix[i][j] = 0;
				}
			} else {
				for (int j = 0; j < columnNum; j++) {
					if (column[j] == 1) {
						matrix[i][j] = 0;
					}
				}
			}
		}
	}
	
	public static void testSetZeros() {
		System.out.println("1.7 setZeros()...");
		int[][] matrix = {{11, 12, 13, 14}, 
				  		  {21, 0, 23, 24}, 
				  		  {31, 32, 33, 34}, 
				  		  {41, 42, 43, 0}};
		System.out.println("Old matrix:\n" + visualizeMatrix(matrix));
		setZeros(matrix);
		System.out.println("New matrix:\n" + visualizeMatrix(matrix));
		System.out.println();
	}

	
	/**
	 * 1.8 Check if s2 is a rotation of s1 using only one call of isSubstring()
	 * @param s1
	 * @param s2
	 */
	public static boolean isRotation(String s1, String s2) {
		if (s1.length() == s2.length()) {
			return isSubstring((s1 + s1), s2);
			
		} else {
			return false;
		}
	}
	
	public static boolean isSubstring(String s1, String s2) {
		int l1 = s1.length();
		int l2 = s2.length();
		for (int i = 0; i < l1 && l2 <= l1 - i; i++) {
			int k = i;
			int j;
			for (j = 0; j < l2; j++) {
				if (s1.charAt(k++) != s2.charAt(j)) break;
			}
			if (j == l2) return true;
		}
		return false;
	}
	
	public static void testIsRotation() {
		System.out.println("1.8 isRotation()...");
		String s1 = "I am Guangcheng Lu.";
		String s2 = "cheng Lu.I am Guang";
		System.out.println("s1: " + s1);
		System.out.println("s2: " + s2);
		System.out.println("s2 is a rotation of s1: " + isRotation(s1, s2));
		System.out.println();
	}
	
	
	public static void testAll() {
		testIsUniqueChars();
		System.out.println("1.2 is in 1.2.cpp.\n");
		testRemoveDuplicate();
		testCheckPermutation();
		testReplaceSpace();
		testSimpleStringCompression();
		testRotateImageMatrix();
		testSetZeros();
		testIsRotation();
	}

	public static void main(String[] args) {
		testAll();
	}
}
