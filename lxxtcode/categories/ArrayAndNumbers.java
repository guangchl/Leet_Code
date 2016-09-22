package categories;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

import org.junit.Test;
import org.junit.Assert;

/**
 * Arrays and Numbers.
 *
 * @author Guangcheng Lu
 */
public class ArrayAndNumbers {

    // ---------------------------------------------------------------------- //
    // ------------------------------- MODELS ------------------------------- //
    // ---------------------------------------------------------------------- //

    /** Definition for an interval. */
    public class Interval {
        int start;
        int end;
        Interval() { start = 0; end = 0; }
        Interval(int s, int e) { start = s; end = e; }
    }

    // ---------------------------------------------------------------------- //
    // ------------------------------ PROBLEMS ------------------------------ //
    // ---------------------------------------------------------------------- //

    /**
     * Plus One.
     *
     * Given a non-negative number represented as an array of digits, plus one
     * to the number. The digits are stored such that the most significant digit
     * is at the head of the list.
     *
     * Example: Given [1,2,3] which represents 123, return [1,2,4]. Given
     * [9,9,9] which represents 999, return [1,0,0,0].
     *
     * @param digits
     *            a number represented as an array of digits
     * @return the result
     */
    @tags.Array
    @tags.Math
    @tags.Company.Google
    @tags.Status.OK
    public int[] plusOne(int[] digits) {
        ArrayList<Integer> list = new ArrayList<>();
        int carry = 1;

        for (int i = digits.length - 1; i >= 0; i--) {
            int digit = digits[i];
            digit += carry;
            carry = digit > 9 ? digit / 10 : 0;

            digit = digit % 10;
            list.add(digit);
        }

        if (carry > 0) {
            list.add(carry);
        }

        int[] result = new int[list.size()];
        for (int i = list.size() - 1, j = 0; i >= 0; i--) {
            result[j++] = list.get(i);
        }

        return result;
    }

    /** Plus One - absolutely awesome solution. */
    @tags.Array
    @tags.Math
    @tags.Company.Google
    @tags.Status.NeedPractice
    public int[] plusOne2(int[] digits) {
        for (int i = digits.length - 1; i >= 0; i--) {
            if (digits[i] < 9) {
                digits[i]++;
                return digits;
            } else {
                digits[i] = 0;
            }
        }

        int[] result = new int[digits.length + 1];
        result[0] = 1;

        return result;
    }

    /**
     * Intersection of Two Arrays
     *
     * Given two arrays, write a function to compute their intersection.
     *
     * Notice: Each element in the result must be unique. The result can be in
     * any order.
     *
     * Example: Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2].
     *
     * @param nums1
     *            an integer array
     * @param nums2
     *            an integer array
     * @return an integer array
     */
    @tags.BinarySearch
    @tags.TwoPointers
    @tags.Sort
    @tags.Array
    @tags.HashTable
    public int[] intersection(int[] nums1, int[] nums2) {
        if (nums1 == null || nums2 == null) {
            return new int[0];
        }

        Set<Integer> set1 = new HashSet<>();
        for (Integer i : nums1) {
            set1.add(i);
        }

        Set<Integer> intersection = new HashSet<>();
        for (Integer i : nums2) {
            if (set1.contains(i)) {
                intersection.add(i);
            }
        }

        int[] result = new int[intersection.size()];
        int index = 0;
        for (Integer num : intersection) {
            result[index++] = num;
        }

        return result;
    }

    /**
     * Intersection of Two Arrays II
     *
     * Given two arrays, write a function to compute their intersection.
     *
     * Notice: Each element in the result should appear as many times as it
     * shows in both arrays. The result can be in any order.
     *
     * Example: Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2, 2].
     *
     * @param nums1
     *            an integer array
     * @param nums2
     *            an integer array
     * @return an integer array
     */
    @tags.BinarySearch
    @tags.TwoPointers
    @tags.Sort
    @tags.Array
    @tags.HashTable
    public int[] intersectionII(int[] nums1, int[] nums2) {
        if (nums1 == null || nums2 == null) {
            return new int[0];
        }

        Map<Integer, Integer> map1 = new HashMap<>();
        for (Integer i : nums1) {
            if (map1.containsKey(i)) {
                map1.put(i, map1.get(i) + 1);
            } else {
                map1.put(i, 1);
            }
        }

        List<Integer> list = new ArrayList<>();
        for (Integer i : nums2) {
            if (map1.containsKey(i)) {
                list.add(i);
                int count = map1.get(i) - 1;
                if (count == 0) {
                    map1.remove(i);
                } else {
                    map1.put(i, count);
                }
            }
        }

        int[] result = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            result[i] = list.get(i);
        }

        return result;
    }

    /**
     * Merge Sorted Array.
     *
     * Given two sorted integer arrays A and B, merge B into A as one sorted
     * array.
     *
     * Notice: You may assume that A has enough space (size that is greater or
     * equal to m + n) to hold additional elements from B. The number of
     * elements initialized in A and B are m and n respectively.
     *
     * Example: A = [1, 2, 3, empty, empty], B = [4, 5] After merge, A will be
     * filled as [1, 2, 3, 4, 5]
     *
     * @param A:
     *            sorted integer array A which has m elements, but size of A is
     *            m+n
     * @param B:
     *            sorted integer array B which has n elements
     * @return: void
     */
    @tags.Array
    @tags.TwoPointers
    @tags.SortedArray
    @tags.Company.Bloomberg
    @tags.Company.Facebook
    @tags.Company.Microsoft
    @tags.Status.OK
    public void mergeSortedArray(int[] A, int m, int[] B, int n) {
        if (A == null || B == null || A.length < m + n) {
            return;
        }

        while (m > 0 && n > 0) {
            A[m + n - 1] = A[m - 1] > B[n - 1] ? A[--m] : B[--n];
        }

        // if numbers left in nums2
        while (n > 0) {
            A[n - 1] = B[n - 1];
            n--;
        }
    }

    /**
     * Merge Two Sorted Arrays
     *
     * Merge two given sorted integer array A and B into a new sorted integer
     * array.
     *
     * Example: A=[1,2,3,4] B=[2,4,5,6] return [1,2,2,3,4,4,5,6]
     *
     * How can you optimize your algorithm if one array is very large and the
     * other is very small?
     *
     * @param A
     *            and B: sorted integer array A and B.
     * @return: A new sorted integer array
     */
    @tags.Array
    @tags.SortedArray
    public int[] mergeSortedArray(int[] A, int[] B) {
        int m = A.length;
        int n = B.length;
        int[] result = new int[m + n];

        int i = 0, j = 0;
        while (i < m || j < n) {
            if (j == n || (i < m && A[i] <= B[j])) {
                result[i + j] = A[i++];
            } else {
                result[i + j] = B[j++];
            }
        }

        return result;
    }

    /**
     * Product of Array Exclude Itself (Product of Array Except Self).
     *
     * Given an integers array A. Define B[i] = A[0] * ... * A[i-1] * A[i+1] *
     * ... * A[n-1], calculate B WITHOUT divide operation.
     *
     * Example: For A = [1, 2, 3], return [6, 3, 2].
     *
     * @param A:
     *            Given an integers array A
     * @return: A Long array B and B[i]= A[0] * ... * A[i-1] * A[i+1] * ... *
     *          A[n-1]
     */
    @tags.Array
    @tags.ForwardBackwardTraversal
    @tags.Source.LintCode
    @tags.Source.LeetCode
    @tags.Company.Amazon
    @tags.Company.Apple
    @tags.Company.Facebook
    @tags.Company.LinkedIn
    @tags.Company.Microsoft
    @tags.Status.NeedPractice
    public ArrayList<Long> productExcludeItself(ArrayList<Integer> A) {
        if (A == null || A.size() == 0) {
            return new ArrayList<>();
        }

        int n = A.size();

        long[] before = new long[n]; // product before i
        before[0] = 1;
        for (int i = 1; i < n; i++) {
            before[i] = before[i - 1] * A.get(i - 1);
        }

        long[] after = new long[n]; // product after i
        after[n - 1] = 1;
        for (int i = n - 2; i >= 0; i--) {
            after[i] = after[i + 1] * A.get(i + 1);
        }

        ArrayList<Long> result = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            result.add(before[i] * after[i]);
        }
        return result;
    }

    /**
     * First Missing Positive.
     *
     * Given an unsorted integer array, find the first missing positive integer.
     *
     * Example: Given [1,2,0] return 3, and [3,4,-1,1] return 2.
     *
     * Challenge: Your algorithm should run in O(n) time and uses constant
     * space.
     *
     * @param A:
     *            an array of integers
     * @return: an integer
     */
    @tags.Array
    public int firstMissingPositive(int[] A) {
        int n = A.length;
        for (int i = 0; i < n;) {
            int num = A[i];
            if (num > 0 && num <= n && num != i + 1 && num != A[num - 1]) {
                A[i] = A[num - 1];
                A[num - 1] = num;
            } else {
                i++;
            }
        }
        for (int i = 0; i < n; i++) {
            if (A[i] != i + 1) {
                return i + 1;
            }
        }
        return n + 1;
    }

    /**
     * Game of Life.
     *
     * According to the Wikipedia's article: "The Game of Life, also known
     * simply as Life, is a cellular automaton devised by the British
     * mathematician John Horton Conway in 1970."
     *
     * Given a board with m by n cells, each cell has an initial state live (1)
     * or dead (0). Each cell interacts with its eight neighbors (horizontal,
     * vertical, diagonal) using the following four rules (taken from the above
     * Wikipedia article):
     *
     * Any live cell with fewer than two live neighbors dies, as if caused by
     * under-population. Any live cell with two or three live neighbors lives on
     * to the next generation. Any live cell with more than three live neighbors
     * dies, as if by over-population.. Any dead cell with exactly three live
     * neighbors becomes a live cell, as if by reproduction.
     *
     * Write a function to compute the next state (after one update) of the
     * board given its current state.
     *
     * Follow up: Could you solve it in-place? Remember that the board needs to
     * be updated at the same time: You cannot update some cells first and then
     * use their updated values to update other cells. In this question, we
     * represent the board using a 2D array. In principle, the board is
     * infinite, which would cause problems when the active area encroaches the
     * border of the array. How would you address these problems?
     *
     * @param board
     */
    @tags.Array
    @tags.Company.Dropbox
    @tags.Company.Google
    @tags.Company.Snapchat
    public void gameOfLife(int[][] board) {
        if (board == null || board.length == 0 || board[0].length == 0) {
            return;
        }

        int m = board.length, n = board[0].length;
        int live = 1, dead = 0, willDie = -1, willLive = 2;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                // count live neighbors
                int liveNeighbor = 0;
                int[] x = { -1, -1, -1, 0, 0, 1, 1, 1 };
                int[] y = { -1, 0, 1, -1, 1, -1, 0, 1 };
                for (int k = 0; k < 8; k++) {
                    int xx = i + x[k], yy = j + y[k];
                    if (xx >= 0 && xx < m && yy >= 0 && yy < n) {
                        if (board[xx][yy] == live || board[xx][yy] == willDie) {
                            liveNeighbor++;
                        }
                    }
                }

                // mark this cell for next gen
                if (board[i][j] == live
                        && (liveNeighbor > 3 || liveNeighbor < 2)) {
                    board[i][j] = willDie;
                } else if (board[i][j] == dead && liveNeighbor == 3) {
                    board[i][j] = willLive;
                }
            }
        }

        // settle down to next gen
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == willLive) {
                    board[i][j] = live;
                } else if (board[i][j] == willDie) {
                    board[i][j] = dead;
                }
            }
        }
    }

    /**
     * Spiral Matrix.
     *
     * Given a matrix of m x n elements (m rows, n columns), return all elements
     * of the matrix in spiral order.
     *
     * Example: Given the following matrix: [ [ 1, 2, 3 ], [ 4, 5, 6 ], [ 7, 8,
     * 9 ] ] You should return [1,2,3,6,9,8,7,4,5].
     *
     * @param matrix
     *            a matrix of m x n elements
     * @return an integer list
     */
    @tags.Array
    @tags.Matrix
    @tags.Company.Google
    @tags.Company.Microsoft
    @tags.Company.Uber
    @tags.Status.Hard
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> result = new ArrayList<>();
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return result;
        }

        int m = matrix.length, n = matrix[0].length;
        int iMin = 0, iMax = m - 1, jMin = 0, jMax = n - 1;
        int i = 0, j = 0, direction = 0;

        while (result.size() < m * n - 1) {
            switch (direction) {
            case 0: // right
                if (j + 1 <= jMax) {
                    result.add(matrix[i][j]);
                    j++;
                } else {
                    iMin = i + 1;
                    direction = (direction + 1) % 4;
                }
                break;
            case 1: // down
                if (i + 1 <= iMax) {
                    result.add(matrix[i][j]);
                    i++;
                } else {
                    jMax = j - 1;
                    direction = (direction + 1) % 4;
                }
                break;
            case 2: // left
                if (j - 1 >= jMin) {
                    result.add(matrix[i][j]);
                    j--;
                } else {
                    iMax = i - 1;
                    direction = (direction + 1) % 4;
                }
                break;
            case 3: // up
                if (i - 1 >= iMin) {
                    result.add(matrix[i][j]);
                    i--;
                } else {
                    jMin = j + 1;
                    direction = (direction + 1) % 4;
                }
                break;
            }
        }

        result.add(matrix[i][j]);
        return result;
    }

    /** Spiral Matrix - another shorter solution. */
    public List<Integer> spiralOrder2(int[][] matrix) {
        List<Integer> result = new ArrayList<>();
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return result;
        }
        int m = matrix.length, n = matrix[0].length;
        int loop = 0;

        while (result.size() < m * n) {
            // top side
            for (int i = loop, j = loop; j < n - loop; j++) {
                result.add(matrix[i][j]);
            }

            // right side
            for (int i = loop + 1, j = n - loop - 1; i < m - loop; i++) {
                result.add(matrix[i][j]);
            }

            // if only one row or col remains, necessary when m != n
            if (m - 2 * loop == 1 || n - 2 * loop == 1) {
                break;
            }

            // bottom side
            for (int i = m - loop - 1, j = n - loop - 2; j >= loop; j--) {
                result.add(matrix[i][j]);
            }

            // left side
            for (int i = m - loop - 2, j = loop; i > loop; i--) {
                result.add(matrix[i][j]);
            }

            loop++;
        }

        return result;
    }

    /**
     * Spiral Matrix II.
     *
     * Given an integer n, generate a square matrix filled with elements from 1
     * to n^2 in spiral order.
     *
     * Example: Given n = 3, You should return the following matrix: [ [ 1, 2, 3
     * ], [ 8, 9, 4 ], [ 7, 6, 5 ] ].
     *
     * @param n
     *            an integer
     * @return a square matrix
     */
    @tags.Array
    public int[][] generateMatrix(int n) {
        int[][] matrix = new int[n][n];
        int loop = 0, num = 1;

        while (num <= n * n) {
            // top
            for (int i = loop, j = loop; j < n - loop; j++) {
                matrix[i][j] = num++;
            }

            // right
            for (int i = loop + 1, j = n - loop - 1; i < n - loop; i++) {
                matrix[i][j] = num++;
            }

            // bottom
            for (int i = n - loop - 1, j = n - loop - 2; j >= loop; j--) {
                matrix[i][j] = num++;
            }

            // left
            for (int i = n - loop - 2, j = loop; i >= loop + 1; i--) {
                matrix[i][j] = num++;
            }

            loop++;
        }

        return matrix;
    }

    /**
     * Reverse Pairs.
     *
     * For an array A, if i < j, and A [i] > A [j], called (A [i], A [j]) is a
     * reverse pair. return total of reverse pairs in A.
     *
     * Example: Given A = [2, 4, 1, 3, 5] , (2, 1), (4, 1), (4, 3) are reverse
     * pairs. return 3
     *
     * @param A
     *            an array
     * @return total of reverse pairs
     */
    @tags.Array
    @tags.MergeSort
    @tags.Status.Hard
    public long reversePairs(int[] A) {
        return mergeSort(A, 0, A.length - 1);
    }

    private int mergeSort(int[] A, int start, int end) {
        if (start >= end) { // > just for empty input
            return 0;
        }

        int mid = (start + end) >>> 1;
        int count = mergeSort(A, start, mid);
        count += mergeSort(A, mid + 1, end);
        count += merge(A, start, mid + 1, end);
        return count;
    }

    private int merge(int[] A, int start1, int start2, int end) {
        int count = 0;
        int[] temp = new int[end - start1 + 1];

        int i = 0, i1 = start1, i2 = start2;
        while (i1 < start2 && i2 <= end) {
            if (A[i1] <= A[i2]) {
                temp[i++] = A[i1++];
            } else {
                count += start2 - i1;
                temp[i++] = A[i2++];
            }
        }

        while (i1 < start2) {
            temp[i++] = A[i1++];
        }
        while (i2 <= end) {
            temp[i++] = A[i2++];
        }

        for (i = 0; i < temp.length; i++) {
            A[start1++] = temp[i];
        }

        return count;
    }

    /**
     * Set Matrix Zeroes.
     *
     * Given a m x n matrix, if an element is 0, set its entire row and column
     * to 0. Do it in place.
     *
     * Example: Given a matrix [ [1,2], [0,3] ], return [ [0,2], [0,0] ].
     *
     * Challenge: Did you use extra space? A straight forward solution using
     * O(mn) space is probably a bad idea. A simple improvement uses O(m + n)
     * space, but still not the best solution. Could you devise a constant space
     * solution?
     *
     * @param matrix:
     *            A list of lists of integers
     * @return: Void
     */
    @tags.Matrix
    @tags.Source.CrackingTheCodingInterview
    @tags.Company.Cloudera
    @tags.Company.Microsoft
    @tags.Status.OK
    public void setZeroes(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return;
        }

        int m = matrix.length, n = matrix[0].length;

        // check first row and column
        boolean firstRow = false, firstCol = false;
        for (int i = 0; i < n; i++) {
            if (matrix[0][i] == 0) {
                firstRow = true;
                break;
            }
        }
        for (int i = 0; i < m; i++) {
            if (matrix[i][0] == 0) {
                firstCol = true;
                break;
            }
        }

        // traverse and mark
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = matrix[0][j] = 0;
                }
            }
        }

        // set 0
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[0][j] == 0 || matrix[i][0] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }

        // mark first row and first column
        if (firstRow) {
            for (int i = 0; i < n; i++) {
                matrix[0][i] = 0;
            }
        }
        if (firstCol) {
            for (int i = 0; i < m; i++) {
                matrix[i][0] = 0;
            }
        }
    }

    /**
     * Sparse Matrix Multiplication.
     *
     * Given two sparse matrices A and B, return the result of AB.
     *
     * You may assume that A's column number is equal to B's row number.
     *
     * @param A
     * @param B
     * @return
     */
    @tags.HashTable
    @tags.Company.Facebook
    @tags.Company.LinkedIn
    public int[][] multiply(int[][] A, int[][] B) {
        if (A == null || B == null) {
            return null;
        }

        int mA = A.length, nA = A[0].length;
        int mB = B.length, nB = B[0].length;
        Map<Integer, Set<Integer>> rowToCol = new HashMap<>();
        Map<Integer, Set<Integer>> colToRow = new HashMap<>();

        // record non-zero columns for each row
        for (int i = 0; i < mA; i++) {
            rowToCol.put(i, new HashSet<Integer>());
            for (int j = 0; j < nA; j++) {
                if (A[i][j] != 0) {
                    rowToCol.get(i).add(j);
                }
            }
        }

        // record non-zero rows for each column
        for (int j = 0; j < nB; j++) {
            colToRow.put(j, new HashSet<Integer>());
            for (int i = 0; i < mB; i++) {
                if (B[i][j] != 0) {
                    colToRow.get(j).add(i);
                }
            }
        }

        // multiply two matrix
        int[][] result = new int[mA][nB];
        for (int i = 0; i < mA; i++) {
            for (int j = 0; j < nB; j++) {
                for (Integer col : rowToCol.get(i)) {
                    if (colToRow.get(j).contains(col)) {
                        result[i][j] += A[i][col] * B[col][j];
                    }
                }
            }
        }

        return result;
    }

    /**
     * Find the Celebrity.
     *
     * Suppose you are at a party with n people (labeled from 0 to n - 1) and
     * among them, there may exist one celebrity. The definition of a celebrity
     * is that all the other n - 1 people know him/her but he/she does not know
     * any of them.
     *
     * Now you want to find out who the celebrity is or verify that there is not
     * one. The only thing you are allowed to do is to ask questions like: "Hi,
     * A. Do you know B?" to get information of whether A knows B. You need to
     * find out the celebrity (or verify there is not one) by asking as few
     * questions as possible (in the asymptotic sense).
     *
     * You are given a helper function bool knows(a, b) which tells you whether
     * A knows B. Implement a function int findCelebrity(n), your function
     * should minimize the number of calls to knows.
     *
     * Note: There will be exactly one celebrity if he/she is in the party.
     * Return the celebrity's label if there is a celebrity in the party. If
     * there is no celebrity, return -1.
     *
     * @param n
     * @return
     */
    @tags.Array
    @tags.Matrix
    @tags.Company.Facebook
    @tags.Company.LinkedIn
    public int findCelebrity(int n) {
        // everyone is possible
        Set<Integer> possible = new HashSet<>();
        for (int i = 0; i < n; i++) {
            possible.add(i);
        }

        // loop through every comibnation
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // no possible one, return -1
                if (possible.isEmpty()) {
                    return -1;
                }

                // skip diagnal and impossible pairs
                if (i != j && (possible.contains(i) || possible.contains(j))) {
                    if (knows(i, j)) {
                        // i knows j means i is not celebrity
                        possible.remove(i);
                    } else {
                        // i doesn't know j means j is not celebrity
                        possible.remove(j);
                    }
                }
            }
        }

        // expect single result only
        if (possible.size() != 1) {
            return -1;
        }

        return possible.iterator().next();
    }

    // Created for compilation
    private boolean knows(int a, int b) {
        return false;
    }

    /**
     * Rotate Array.
     *
     * Rotate an array of n elements to the right by k steps.
     *
     * For example, with n = 7 and k = 3, the array [1,2,3,4,5,6,7] is rotated
     * to [5,6,7,1,2,3,4].
     *
     * Note: Try to come up as many solutions as you can, there are at least 3
     * different ways to solve this problem.
     *
     * Hint: Could you do it in-place with O(1) extra space?
     *
     * @param nums
     * @param k
     */
    @tags.Array
    @tags.Company.Bloomberg
    @tags.Company.LinkedIn
    @tags.Company.Microsoft
    public void rotate(int[] nums, int k) {
        int n = nums.length;
        k = k % n;
        if (k != 0) {
           reverse(nums, 0, n - k - 1);
            reverse(nums, n - k, n - 1);
            reverse(nums, 0, n - 1);
        }
    }

    private void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int tmp = nums[start];
            nums[start] = nums[end];
            nums[end] = tmp;
            start++;
            end--;
        }
    }

    // ---------------------------------------------------------------------- //
    // ----------------------- Shortest Word Distance ----------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Shortest Word Distance.
     *
     * Given a list of words and two words word1 and word2, return the shortest
     * distance between these two words in the list.
     *
     * For example, Assume that words = ["practice", "makes", "perfect",
     * "coding", "makes"]. Given word1 = ¡°coding¡±, word2 = ¡°practice¡±, return 3.
     * Given word1 = "makes", word2 = "coding", return 1.
     *
     * Note: You may assume that word1 does not equal to word2, and word1 and
     * word2 are both in the list.
     *
     * @param words
     * @param word1
     * @param word2
     * @return
     */
    @tags.Array
    @tags.Company.LinkedIn
    @tags.Status.NeedPractice
    public int shortestDistance(String[] words, String word1, String word2) {
        if (words == null || words.length == 0) {
            throw new IllegalArgumentException();
        }

        int n = words.length;
        int pos1 = findWord(words, word1, 0);
        int pos2 = findWord(words, word2, 0);
        int min = n - 1;

        while (true) {
            min = Math.min(min, Math.abs(pos1 - pos2));
            if (pos1 < pos2) {
                int next = findWord(words, word1, pos1 + 1);
                if (next != n) {
                    pos1 = next;
                    continue;
                } else {
                    break;
                }
            }
            if (pos2 < pos1) {
                int next = findWord(words, word2, pos2 + 1);
                if (next != n) {
                    pos2 = next;
                    continue;
                } else {
                    break;
                }
            }
        }

        return min;
    }

    private int findWord(String[] words, String word, int pos) {
        for (int i = pos; i < words.length; i++) {
            if (words[i].equals(word)) {
                return i;
            }
        }
        return words.length;
    }

    /**
     * Shortest Word Distance II.
     *
     * This is a follow up of Shortest Word Distance. The only difference is now
     * you are given the list of words and your method will be called repeatedly
     * many times with different parameters. How would you optimize it? Design a
     * class which receives a list of words in the constructor, and implements a
     * method that takes two words word1 and word2 and return the shortest
     * distance between these two words in the list.
     *
     * For example, Assume that words = ["practice", "makes", "perfect",
     * "coding", "makes"].
     *
     * Given word1 = ¡°coding¡±, word2 = ¡°practice¡±, return 3. Given word1 =
     * "makes", word2 = "coding", return 1.
     *
     * Note: You may assume that word1 does not equal to word2, and word1 and
     * word2 are both in the list.
     */
    @tags.Design
    @tags.HashTable
    @tags.Company.LinkedIn
    @tags.Status.NeedPractice
    public class WordDistance {
        // Your WordDistance object will be instantiated and called as such:
        // WordDistance wordDistance = new WordDistance(words);
        // wordDistance.shortest("word1", "word2");
        // wordDistance.shortest("anotherWord1", "anotherWord2");
        Map<String, List<Integer>> map = new HashMap<>();

        public WordDistance(String[] words) {
            for (int i = 0; i < words.length; i++) {
                if (map.containsKey(words[i])) {
                    map.get(words[i]).add(i);
                } else {
                    List<Integer> list = new ArrayList<>();
                    list.add(i);
                    map.put(words[i], list);
                }
            }
        }

        /**
         * You can further cache the result of this method.
         */
        public int shortest(String word1, String word2) {
            List<Integer> list1 = map.get(word1), list2 = map.get(word2);
            int pos1 = 0, pos2 = 0;
            int min = Math.abs(list1.get(0) - list2.get(0));

            while (pos1 < list1.size() && pos2 < list2.size()) {
                min = Math.min(min, Math.abs(list1.get(pos1) - list2.get(pos2)));
                if (list1.get(pos1) < list2.get(pos2)) {
                    pos1++;
                } else {
                    pos2++;
                }
            }

            return min;
        }
    }

    /**
     * Shortest Word Distance III.
     *
     * This is a follow up of Shortest Word Distance. The only difference is now
     * word1 could be the same as word2. Given a list of words and two words
     * word1 and word2, return the shortest distance between these two words in
     * the list. word1 and word2 may be the same and they represent two
     * individual words in the list.
     *
     * For example, Assume that words = ["practice", "makes", "perfect",
     * "coding", "makes"].
     *
     * Given word1 = ¡°makes¡±, word2 = ¡°coding¡±, return 1. Given word1 = "makes",
     * word2 = "makes", return 3.
     *
     * Note: You may assume word1 and word2 are both in the list.
     *
     * @param words
     * @param word1
     * @param word2
     * @return
     */
    @tags.Array
    @tags.Company.LinkedIn
    @tags.Status.NeedPractice
    public int shortestWordDistance(String[] words, String word1, String word2) {
        int pos1 = findWord(words, word1, 0, -1);
        int pos2 = findWord(words, word2, 0, pos1);
        int n = words.length;
        int min = n - 1;

        while (pos1 < n && pos2 < n) {
            System.out.println(pos1 + " " + pos2);
            min = Math.min(min, Math.abs(pos1 - pos2));
            if (pos1 < pos2) {
                pos1 = findWord(words, word1, pos1 + 1, pos2);
            } else {
                pos2 = findWord(words, word2, pos2 + 1, pos1);
            }
        }

        return min;
    }

    private int findWord(String[] words, String word, int pos, int except) {
        for (int i = pos; i < words.length; i++) {
            if (i == except)
                continue;
            if (words[i].equals(word)) {
                return i;
            }
        }
        return words.length;
    }

    // ---------------------------------------------------------------------- //
    // ------------------------------ H-Index ------------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * H-Index.
     *
     * Given an array of citations (each citation is a non-negative integer) of
     * a researcher, write a function to compute the researcher's h-index.
     *
     * According to the definition of h-index on Wikipedia: "A scientist has
     * index h if h of his/her N papers have at least h citations each, and the
     * other N - h papers have no more than h citations each."
     *
     * For example, given citations = [3, 0, 6, 1, 5], which means the
     * researcher has 5 papers in total and each of them had received 3, 0, 6,
     * 1, 5 citations respectively. Since the researcher has 3 papers with at
     * least 3 citations each and the remaining two with no more than 3
     * citations each, his h-index is 3.
     *
     * Note: If there are several possible values for h, the maximum one is
     * taken as the h-index.
     *
     * Hint: An easy approach is to sort the array first.
     *
     * @param citations
     * @return
     */
    @tags.Sort
    @tags.Company.Bloomberg
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Status.NeedPractice
    public int hIndex(int[] citations) {
        // nlogn solution

        Arrays.sort(citations);
        int h = 0;

        for (int i = 0; i < citations.length; i++) {
            int countOrCitation = Math.min(citations.length - i, citations[i]);
            h = Math.max(h, countOrCitation);
        }

        return h;
    }

    /** H-Index. */
    @tags.HashTable
    @tags.Company.Bloomberg
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Status.NeedPractice
    public int hIndex2(int[] citations) {
        // O(n) time and O(n) space

        int n = citations.length;
        int[] citationCount = new int[n + 1];
        for (int i = 0; i < n; i++) {
            citationCount[citations[i] > n ? n : citations[i]]++;
        }

        int sum = 0;
        for (int i = n; i >= 1; i--) {
            sum += citationCount[i];
            if (sum >= i) {
                return i;
            }
        }
        return 0;
    }

    /**
     * H-Index II.
     *
     * Follow up for H-Index: What if the citations array is sorted in ascending
     * order? Could you optimize your algorithm?
     *
     * Hint: Expected runtime complexity is in O(log n) and the input is sorted.
     *
     * @param citations
     * @return
     */
    @tags.BinarySearch
    @tags.Company.Facebook
    public int hIndexII(int[] citations) {
        // O(n) time

        int n = citations.length;
        int start = 0, end = n - 1;
        while (start <= end) {
            int mid = (start + end) >>> 1;
            int count = n - mid;
            int citation = citations[mid];
            if (count > citation) {
                start = mid + 1;
            } else if (count < citation) {
                end = mid - 1;
            } else {
                return count;
            }
        }
        return n - start;
    }

    // ---------------------------------------------------------------------- //
    // ----------------------------- INTERVALS ------------------------------ //
    // ---------------------------------------------------------------------- //

    /**
     * Merge Intervals.
     *
     * Given a collection of intervals, merge all overlapping intervals.
     *
     * Example: Given intervals => merged intervals:
     * [[2,3],[2,2],[3,3],[1,3],[5,7],[2,2],[4,6]] => [[1,3],[4,7]].
     *
     * Challenge: O(n log n) time and O(1) extra space.
     *
     * @param intervals,
     *            a collection of intervals
     * @return: A new sorted interval list.
     */
    @tags.Array
    @tags.Sort
    @tags.Company.Bloomberg
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Company.LinkedIn
    @tags.Company.Microsoft
    @tags.Company.Twitter
    @tags.Company.Yelp
    @tags.Status.NeedPractice
    public List<Interval> merge(List<Interval> intervals) {
        List<Interval> result = new ArrayList<>();
        if (intervals == null || intervals.size() == 0) {
            return result;
        }

        Collections.sort(intervals, new Comparator<Interval>() {
            @Override
            public int compare(Interval i1, Interval i2) {
                return i1.start - i2.start;
            }
        });

        Interval current = intervals.get(0);
        current = new Interval(current.start, current.end);
        for (int i = 1; i < intervals.size(); i++) {
            if (current.end >= intervals.get(i).start) {
                current.end = Math.max(current.end, intervals.get(i).end);
            } else {
                result.add(current);
                current = intervals.get(i);
            }
        }

        result.add(current);
        return result;
    }

    /**
     * Summary Ranges.
     *
     * Given a sorted integer array without duplicates, return the summary of
     * its ranges.
     *
     * For example, given [0,1,2,4,5,7], return ["0->2","4->5","7"].
     *
     * @param nums
     * @return
     */
    @tags.Array
    @tags.Company.Google
    @tags.Status.OK
    public List<String> summaryRanges(int[] nums) {
        List<String> result = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            return result;
        }

        int start = nums[0], end = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] == end + 1) {
                end++;
            } else {
                result.add(rangeToString(start, end));

                start = nums[i];
                end = nums[i];
            }
        }

        result.add(rangeToString(start, end));

        return result;
    }

    private String rangeToString(int start, int end) {
        if (start == end) {
            return String.valueOf(start);
        }

        StringBuilder sb = new StringBuilder();
        sb.append(start);
        sb.append("->");
        sb.append(end);
        return sb.toString();
    }

    /**
     * Insert Interval.
     *
     * Given a non-overlapping interval list which is sorted by start point.
     * Insert a new interval into it, make sure the list is still in order and
     * non-overlapping (merge intervals if necessary).
     *
     * Example: Insert [2, 5] into [[1,2], [5,9]], we get [[1,9]]. Insert [3, 4]
     * into [[1,2], [5,9]], we get [[1,2], [3,4], [5,9]].
     *
     * @param intervals:
     *            Sorted interval list.
     * @param newInterval:
     *            A new interval.
     * @return: A new sorted interval list.
     */
    @tags.BasicImplementation
    @tags.Array
    @tags.Sort
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Company.LinkedIn
    @tags.Status.NeedPractice
    public ArrayList<Interval> insert(ArrayList<Interval> intervals,
            Interval newInterval) {
        ArrayList<Interval> result = new ArrayList<Interval>();

        for (Interval interval : intervals) {
            if (newInterval == null || interval.end < newInterval.start) {
                result.add(interval);
            } else {
                if (interval.start > newInterval.end) {
                    result.add(newInterval);
                    result.add(interval);
                    newInterval = null;
                } else {
                    newInterval.start = Math.min(interval.start,
                            newInterval.start);
                    newInterval.end = Math.max(interval.end, newInterval.end);
                }
            }
        }

        if (newInterval != null) {
            result.add(newInterval);
        }

        return result;
    }

    /**
     * Missing Ranges.
     *
     * Given a sorted integer array where the range of elements are [lower,
     * upper] inclusive, return its missing ranges.
     *
     * For example, given [0, 1, 3, 50, 75], lower = 0 and upper = 99, return
     * ["2", "4->49", "51->74", "76->99"].
     *
     * @param nums
     * @param lower
     * @param upper
     * @return
     */
    @tags.Array
    @tags.Company.Google
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        List<String> ranges = new ArrayList<>();
        if (nums == null) {
            return ranges;
        }

        for (int i = 0; i < nums.length; i++) {
            if (lower < nums[i]) {
                ranges.add(rangeToString(lower, nums[i] - 1));
            }
            lower = nums[i] + 1;
        }

        if (lower <= upper) {
            ranges.add(rangeToString(lower, upper));
        }

        return ranges;
    }

    /**
     * Number of Airplanes in the Sky.
     *
     * Given an interval list which are flying and landing time of the flight.
     * How many airplanes are on the sky at most?
     *
     * Notice: If landing and flying happens at the same time, we consider
     * landing should happen at first.
     *
     * Example: For interval list [ [1,10], [2,3], [5,8], [4,7] ], Return 3.
     *
     * @param intervals:
     *            An interval array
     * @return: Count of airplanes are in the sky.
     */
    @tags.Array
    @tags.Interval
    @tags.Source.LintCode
    public int countOfAirplanes(List<Interval> airplanes) {
        class Point {
            int time;
            int isEnter;

            public Point(int time, int isEnter) {
                this.time = time;
                this.isEnter = isEnter; // 1 for enter, -1 for exit
            }
        }

        List<Point> list = new ArrayList<>();
        for (Interval i : airplanes) {
            list.add(new Point(i.start, 1));
            list.add(new Point(i.end, -1));
        }
        Collections.sort(list, new Comparator<Point>() {
            @Override
            public int compare(Point p1, Point p2) {
                if (p1.time != p2.time) {
                    return p1.time - p2.time;
                } else {
                    return p1.isEnter - p2.isEnter;
                }
            }
        });

        int count = 0, max = 0;
        for (int i = 0; i < list.size(); i++) {
            count += list.get(i).isEnter;
            max = Math.max(max, count);
        }

        return max;
    }

    /**
     * Meeting Rooms.
     *
     * Given an array of meeting time intervals consisting of start and end
     * times [[s1,e1],[s2,e2],...] (si < ei), determine if a person could attend
     * all meetings.
     *
     * For example, Given [[0, 30],[5, 10],[15, 20]], return false.
     *
     * @param intervals
     * @return
     */
    @tags.Sort
    @tags.Company.Facebook
    public boolean canAttendMeetings(Interval[] intervals) {
        Arrays.sort(intervals, new Comparator<Interval>() {
            @Override
            public int compare(Interval i1, Interval i2) {
                if (i1.start != i2.start) {
                    return i1.start - i2.start;
                }
                return i1.end - i2.end;
            }
        });

        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i].start < intervals[i - 1].end) {
                return false;
            }
        }

        return true;
    }

    /**
     * Meeting Rooms II.
     *
     * Given an array of meeting time intervals consisting of start and end
     * times [[s1,e1],[s2,e2],...] (si < ei), find the minimum number of
     * conference rooms required.
     *
     * For example, Given [[0, 30],[5, 10],[15, 20]], return 2.
     *
     * @param intervals
     * @return
     */
    @tags.Sort
    @tags.Company.Facebook
    public int minMeetingRooms(Interval[] intervals) {
        class Point {
            int time;
            int isStart;

            public Point(int time, int isStart) {
                this.time = time;
                this.isStart = isStart;
            }
        }

        List<Point> points = new ArrayList<>();
        for (Interval i : intervals) {
            points.add(new Point(i.start, 1));
            points.add(new Point(i.end, -1));
        }

        Collections.sort(points, new Comparator<Point>() {
            @Override
            public int compare(Point p1, Point p2) {
                if (p1.time != p2.time) {
                    return p1.time - p2.time;
                }
                return p1.isStart - p2.isStart;
            }
        });

        int max = 0;
        int count = 0;

        for (Point p : points) {
            count += p.isStart;
            max = Math.max(max, count);
        }

        return max;
    }

    // ---------------------------------------------------------------------- //
    // ---------------------- PARTITION, TWO POINTERS ----------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Remove Element.
     *
     * Given an array and a value, remove all occurrences of that value in place
     * and return the new length. The order of elements can be changed, and the
     * elements after the new length don't matter.
     *
     * Example: Given an array [0,4,4,0,0,2,4,4], value=4. return 4 and front
     * four elements of the array is [0,0,0,2].
     *
     * @param A:
     *            A list of integers
     * @param elem:
     *            An integer
     * @return: The new length after remove
     */
    @tags.Array
    @tags.TwoPointers
    public int removeElement(int[] A, int elem) {
        int i = 0, j = A.length - 1;
        while (i <= j) {
            if (A[i] == elem) {
                A[i] = A[j];
                j--;
            } else {
                i++;
            }
        }
        return j + 1;
    }

    /**
     * Remove Duplicates from Sorted Array.
     *
     * Given a sorted array, remove the duplicates in place such that each
     * element appear only once and return the new length.
     *
     * Do not allocate extra space for another array, you must do this in place
     * with constant memory.
     *
     * Example: Given input array A = [1,1,2], Your function should return
     * length = 2, and A is now [1,2].
     *
     * @param A
     * @return new length
     */
    @tags.Array
    @tags.TwoPointers
    @tags.Company.Bloomberg
    @tags.Company.Facebook
    @tags.Company.Microsoft
    public int removeDuplicates(int[] A) {
        if (A == null || A.length == 0) {
            return 0;
        }

        int last = 0;

        for (int i = 1; i < A.length; i++) {
            if (A[i] != A[last]) {
                A[++last] = A[i];
            }
        }

        return last + 1;
    }

    /**
     * Remove Duplicates from Sorted Array II
     *
     * Follow up for "Remove Duplicates": What if duplicates are allowed at most
     * twice?
     *
     * For example, Given sorted array A = [1,1,1,2,2,3], your function should
     * return length = 5, and A is now [1,1,2,2,3].
     *
     * @param A: a array of integers
     * @return : return an integer
     */
    @tags.Array
    @tags.TwoPointers
    @tags.Company.Facebook
    public int removeDuplicatesII(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }

        int n = nums.length, count = 1, last = -1;
        for (int i = 1; i < n; i++) {
            if (nums[i] == nums[i - 1]) {
                count++;
            } else {
                // deal with previous batch
                nums[++last] = nums[i - 1];
                if (count != 1) {
                    nums[++last] = nums[i - 1];
                }

                // start new count
                count = 1;
            }
        }

        // last batch
        nums[++last] = nums[n - 1];
        if (count != 1) {
            nums[++last] = nums[n - 1];
        }
        return last + 1;
    }

    /** Remove Duplicates from Sorted Array II - cleaner solution. */
    @tags.Array
    @tags.TwoPointers
    @tags.Company.Facebook
    public int removeDuplicatesII2(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }

        int last = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[last] || last == 0
                    || nums[i] != nums[last - 1]) {
                nums[++last] = nums[i];
            }
        }
        return last + 1;
    }

    /**
     * Partition Array by Odd and Even.
     *
     * Partition an integers array into odd number first and even number second.
     *
     * Example: Given [1, 2, 3, 4], return [1, 3, 2, 4].
     *
     * Challenge: Do it in-place.
     *
     * @param nums:
     *            an array of integers
     * @return: nothing
     */
    @tags.Array
    @tags.TwoPointers
    @tags.Status.OK
    public void partitionArray(int[] nums) {
        if (nums == null) {
            return;
        }

        for (int start = 0, end = nums.length - 1; start < end; start++, end--) {
            while (nums[start] % 2 == 1) {
                start++;
            }
            while (nums[end] % 2 == 0) {
                end--;
            }
            
            // swap
            if (start < end) {
                int tmp = nums[start];
                nums[start] = nums[end];
                nums[end] = tmp;
            }
        }
    }

    /**
     * Sort Colors
     *
     * Given an array with n objects colored red, white or blue, sort them so
     * that objects of the same color are adjacent, with the colors in the order
     * red, white and blue.
     *
     * Here, we will use the integers 0, 1, and 2 to represent the color red,
     * white, and blue respectively.
     *
     * Note: You are not suppose to use the library's sort function for this
     * problem.
     *
     * Example Given [1, 0, 1, 2], sort it in-place to [0, 1, 1, 2].
     *
     * @param nums:
     *            A list of integer which is 0, 1 or 2
     * @return: nothing
     */
    @tags.TwoPointers
    @tags.Array
    @tags.Sort
    @tags.Company.Facebook
    public void sortColors(int[] nums) {
        if (nums == null) {
            return;
        }

        int left = 0, right = nums.length - 1;
        for (int i = 0; i <= right; i++) {
            if (nums[i] == 0) {
                nums[i] = nums[left];
                nums[left++] = 0;
            } else if (nums[i] == 2) {
                nums[i] = nums[right];
                nums[right--] = 2;
                i--;
            }
        }
    }

    /**
     * Sort Colors II
     *
     * Given an array of n objects with k different colors (numbered from 1 to
     * k), sort them so that objects of the same color are adjacent, with the
     * colors in the order 1, 2, ... k.
     *
     * You are not suppose to use the library's sort function for this problem.
     *
     * Example: Given colors=[3, 2, 2, 1, 4], k=4, your code should sort colors
     * in-place to [1, 2, 2, 3, 4].
     *
     * Challenge: A rather straight forward solution is a two-pass algorithm
     * using counting sort. That will cost O(k) extra memory. Can you do it
     * without using extra memory?
     *
     * This is a decent question with freaking stupid challenge instruction.
     * Best solution is the so called rather straight forward one, counting
     * sort. A close second is using the Arrays.sort. The worst solution is the
     * suggested no extra space one. What's the point? Of course, I understand
     * it always depends.
     *
     * @param colors:
     *            A list of integer
     * @param k:
     *            An integer
     * @return: void
     */
    @tags.Array
    @tags.TwoPointers
    @tags.Sort
    public void sortColors2(int[] colors, int k) {
        if (colors == null || colors.length < 2) {
            return;
        }

        // find the min and max colors
        int min = k, max = 1;
        for (Integer color : colors) {
            if (color > max) {
                max = color;
            } else if (color < min) {
                min = color;
            }
        }

        // each outer loop move min and max to the left and right
        int left = 0, right = colors.length - 1;
        while (left < right) {
            int current = left;
            while (current <= right) {
                if (colors[current] == max) {
                    colors[current] = colors[right];
                    colors[right--] = max;
                } else if (colors[current] == min) {
                    colors[current] = colors[left];
                    colors[left++] = min;
                    current++;
                } else {
                    current++;
                }
            }
            min++;
            max--;
        }
    }

    /**
     * Sort Letters by Case.
     *
     * Given a string which contains only letters. Sort it by lower case first
     * and upper case second.
     *
     * Notice: It's NOT necessary to keep the original order of lower-case
     * letters and upper case letters.
     *
     * Example: For "abAcD", a reasonable answer is "acbAD"
     *
     * @param chars:
     *            The letter array you should sort by Case
     * @return: void
     */
    @tags.Array
    @tags.String
    @tags.Sort
    @tags.TwoPointers
    @tags.Source.LintCode
    public void sortLetters(char[] chars) {
        int len = chars.length;
        for (int left = 0, right = len - 1; left < right;) {
            if (chars[left] >= 'a' && chars[left] <= 'z') {
                left++;
            } else {
                char temp = chars[left];
                chars[left] = chars[right];
                chars[right] = temp;
                right--;
            }
        }
    }

    /**
     * Minimum Window Substring.
     *
     * Given a string S and a string T, find the minimum window in S which will
     * contain all the characters in T in complexity O(n).
     *
     * For example, S = "ADOBECODEBANC" T = "ABC" Minimum window is "BANC".
     *
     * Note: If there is no such window in S that covers all characters in T,
     * return the empty string "".
     *
     * If there are multiple such windows, you are guaranteed that there will
     * always be only one unique minimum window in S.
     *
     * @param s: A string
     * @param t: A string
     * @return: A string denote the minimum window
     *          Return "" if there is no such a string
     */
    @tags.String
    @tags.HashTable
    @tags.TwoPointers
    @tags.Company.Facebook
    @tags.Company.LinkedIn
    @tags.Company.Snapchat
    @tags.Company.Uber
    public String minWindow(String s, String t) {
        Map<Character, Integer> tCount = new HashMap<>();

        // count target as negative numbers
        for (int i = 0; i < t.length(); i++) {
            char c = t.charAt(i);
            if (!tCount.containsKey(c)) {
                tCount.put(c, -1);
            } else {
                tCount.put(c, tCount.get(c) - 1);
            }
        }

        // target is empty
        if (tCount.isEmpty()) {
            return "";
        }

        String window = "";
        Set<Character> less = new HashSet<>(tCount.keySet());
        int start = 0, end = -1;

        while (true) {
            if (less.isEmpty()) { // valid window
                // shorter window
                if (window.length() == 0
                        || (end - start + 1 < window.length())) {
                    window = s.substring(start, end + 1);
                }

                // shrink the start
                char c = s.charAt(start);
                if (tCount.containsKey(c)) {
                    tCount.put(c, tCount.get(c) - 1);
                    if (tCount.get(c) == -1) {
                        less.add(c);
                    }
                }
                start++;
            } else { // lacking chars
                // expand the end
                end++;
                if (end == s.length()) {
                    return window;
                }
                char c = s.charAt(end);
                if (tCount.containsKey(c)) {
                    tCount.put(c, tCount.get(c) + 1);
                    if (tCount.get(c) == 0) {
                        less.remove(c);
                    }
                }
            }
        }
    }

    /**
     * Move Zeroes.
     *
     * Given an array nums, write a function to move all 0's to the end of it
     * while maintaining the relative order of the non-zero elements.
     *
     * For example, given nums = [0, 1, 0, 3, 12], after calling your function,
     * nums should be [1, 3, 12, 0, 0].
     *
     * Note: You must do this in-place without making a copy of the array.
     * Minimize the total number of operations.
     *
     * @param nums
     */
    @tags.Array
    @tags.TwoPointers
    @tags.Company.Bloomberg
    @tags.Company.Facebook
    public void moveZeroes(int[] nums) {
        int last = 0;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                nums[last++] = nums[i];
            }
        }

        for (int i = last; i < nums.length; i++) {
            nums[i] = 0;
        }
    }

    // ---------------------------------------------------------------------- //
    // -------------------------- Partition, Top K -------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Partition Array
     *
     * Given an array nums of integers and an int k, partition the array (i.e
     * move the elements in "nums") such that: All elements < k are moved to the
     * left All elements >= k are moved to the right Return the partitioning
     * index, i.e the first index i nums[i] >= k.
     *
     * You should do really partition in array nums instead of just counting the
     * numbers of integers smaller than k. If all elements in nums are smaller
     * than k, then return nums.length.
     *
     * Example: If nums = [3,2,2,1] and k=2, a valid answer is 1.
     *
     * Challenge: Can you partition the array in-place and in O(n)?
     *
     * @param nums:
     *            The integer array you should partition
     * @param k:
     *            As description return: The index after partition
     */
    @tags.Array
    @tags.TwoPointers
    @tags.Sort
    public int partitionArray(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return 0;
        }

        int start = 0, end = nums.length - 1;
        while (start <= end) {
            if (nums[start] >= k) {
                int tmp = nums[start];
                nums[start] = nums[end];
                nums[end] = tmp;
                end--;
            } else {
                start++;
            }
        }
        return start;
    }

    /**
     * Median.
     *
     * Given a unsorted array with integers, find the median of it. A median is
     * the middle number of the array after it is sorted. If there are even
     * numbers in the array, return the N/2-th number after sorted.
     *
     * Example: Given [4, 5, 1, 2, 3], return 3. Given [7, 9, 4, 5], return 5.
     *
     * Challenge: O(n) time.
     *
     * @param nums:
     *            A list of integers.
     * @return: An integer denotes the middle number of the array.
     */
    @tags.Array
    @tags.QuickSort
    @tags.Source.LintCode
    @tags.Status.NeedPractice
    public int median(int[] nums) {
        int start = 0, end = nums.length - 1;
        int m = (start + end) >>> 1;
        while (true) {
            int pivot = partition(start, end, nums);
            if (pivot < m) {
                start = pivot + 1;
            } else if (pivot > m) {
                end = pivot - 1;
            } else {
                return nums[m];
            }
        }
    }

    private int partition(int start, int end, int[] nums) {
        int pivot = start++;
        while (start <= end) {
            if (nums[start] <= nums[pivot]) {
                start++;
            } else {
                int tmp = nums[start];
                nums[start] = nums[end];
                nums[end] = tmp;
                end--;
            }
        }
        int tmp = nums[pivot];
        nums[pivot] = nums[end];
        nums[end] = tmp;
        return end;
    }

    /**
     * Kth Largest Element.
     *
     * Find K-th largest element in an array.
     *
     * Notice: You can swap elements in the array
     *
     * Example: In array [9,3,2,4,8], the 3rd largest element is 4. In array
     * [1,2,3,4,5], the 1st largest element is 5, 2nd largest element is 4, 3rd
     * largest element is 3 and etc.
     *
     * Challenge: O(n) time, O(1) extra memory. Geometric series amortized O(n)
     * time, which is O(n + n/2 + n/4 + ... + 1).
     *
     * @param k
     *            : description of k
     * @param nums
     *            : array of nums
     * @return: description of return
     */
    @tags.Array
    @tags.Sort
    @tags.QuickSort
    @tags.Heap
    @tags.DivideAndConquer
    @tags.Status.NeedPractice
    public int kthLargestElement(int k, int[] nums) {
        if (nums == null || nums.length < k) {
            throw new IllegalArgumentException();
        }
        return getKthNumber(k, nums, 0, nums.length - 1);
    }

    private int getKthNumber(int k, int[] nums, int start, int end) {
        int pivot = partition(nums, start, end);
        if (pivot < k - 1) {
            return getKthNumber(k, nums, pivot + 1, end);
        } else if (pivot > k - 1) {
            return getKthNumber(k, nums, start, pivot - 1);
        } else {
            return nums[pivot];
        }
    }

    private int partition(int[] nums, int start, int end) {
        int pivot = start;
        while (start <= end) {
            while (start <= end && nums[start] >= nums[pivot]) {
                start++;
            }
            while (start <= end && nums[end] <= nums[pivot]) {
                end--;
            }

            if (start < end) {
                int temp = nums[start];
                nums[start] = nums[end];
                nums[end] = temp;
            }
        }
        int temp = nums[pivot];
        nums[pivot] = nums[end];
        nums[end] = temp;
        return end;
    }

    /**
     * Median of two Sorted Arrays
     *
     * There are two sorted arrays A and B of size m and n respectively. Find
     * the median of the two sorted arrays.
     *
     * Example: Given A=[1,2,3,4,5,6] and B=[2,3,4,5], the median is 3.5. Given
     * A=[1,2,3] and B=[4,5], the median is 3.
     *
     * @param A:
     *            An integer array.
     * @param B:
     *            An integer array.
     * @return: a double whose format is *.5 or *.0
     */
    @tags.Array
    @tags.SortedArray
    @tags.DivideAndConquer
    @tags.BinarySearch
    @tags.Company.Adobe
    @tags.Company.Apple
    @tags.Company.Dropbox
    @tags.Company.Google
    @tags.Company.Microsoft
    @tags.Company.Uber
    @tags.Company.Yahoo
    @tags.Company.Zenefits
    @tags.Status.Hard
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1 == null || nums2 == null
                || nums1.length + nums2.length == 0) {
            return 0;
        }

        int len = nums1.length + nums2.length;

        if (len % 2 == 1) {
            return findKth(nums1, 0, nums2, 0, len / 2 + 1);
        } else {
            int left = findKth(nums1, 0, nums2, 0, len / 2);
            int right = findKth(nums1, 0, nums2, 0, len / 2 + 1);
            return (left + right) / 2.0;
        }
    }

    // k is 1-based, not index
    public int findKth(int[] A, int aStart, int[] B, int bStart, int k) {
        if (aStart == A.length) {
            return B[bStart + k - 1];
        } else if (bStart == B.length) {
            return A[aStart + k - 1];
        }

        if (k == 1) {
            return Math.min(A[aStart], B[bStart]);
        }

        int aNum = aStart + k / 2 - 1 >= A.length ? Integer.MAX_VALUE
                : A[aStart + k / 2 - 1];
        int bNum = bStart + k / 2 - 1 >= B.length ? Integer.MAX_VALUE
                : B[bStart + k / 2 - 1];

        if (aNum <= bNum) {
            return findKth(A, aStart + k / 2, B, bStart, k - k / 2);
        } else {
            return findKth(A, aStart, B, bStart + k / 2, k - k / 2);
        }
    }

    /**
     * Top k Largest Numbers.
     *
     * Given an integer array, find the top k largest numbers in it.
     *
     * Example: Given [3,10,1000,-99,4,100] and k = 3. Return [1000, 100, 10].
     *
     * @param nums
     *            an integer array
     * @param k
     *            an integer
     * @return the top k largest numbers in array
     */
    @tags.PriorityQueue
    @tags.Heap
    public int[] topk(int[] nums, int k) {
        if (nums == null || k < 0) {
            return new int[0];
        }

        PriorityQueue<Integer> pq = new PriorityQueue<>(k,
                new Comparator<Integer>() {
                    @Override
                    public int compare(Integer i1, Integer i2) {
                        return i1 - i2;
                    }
                });
        for (int i = 0; i < nums.length; i++) {
            pq.offer(nums[i]);
            if (pq.size() > k) {
                pq.poll();
            }
        }

        int[] result = new int[k];
        for (int i = result.length - 1; i >= 0; i--) {
            result[i] = pq.poll();
        }
        return result;
    }

    /**
     * Top k Largest Numbers II.
     *
     * Implement a data structure, provide two interfaces: 1.add(number). Add a
     * new number in the data structure. 2.topk(). Return the top k largest
     * numbers in this data structure. k is given when we create the data
     * structure.
     */
    @tags.Heap
    @tags.PriorityQueue
    public class TopKClass {
        private PriorityQueue<Integer> pq;
        private int k;

        public TopKClass(int k) {
            this.k = k;
            pq = new PriorityQueue<>(k, new Comparator<Integer>() {
                @Override
                public int compare(Integer i1, Integer i2) {
                    return i2 - i1;
                }
            });
        }

        public void add(int num) {
            pq.offer(num);
        }

        public List<Integer> topk() {
            List<Integer> result = new ArrayList<>();
            int kk = k;

            while (kk-- > 0 && !pq.isEmpty()) {
                result.add(pq.poll());
            }

            for (Integer i : result) {
                pq.offer(i);
            }

            return result;
        }
    };

    /**
     * Top K Frequent Words.
     *
     * Given a list of words and an integer k, return the top k frequent words
     * in the list.
     *
     * Notice: You should order the words by the frequency of them in the return
     * list, the most frequent one comes first. If two words has the same
     * frequency, the one with lower alphabetical order come first.
     *
     * Example: Given [ "yes", "lint", "code", "yes", "code", "baby", "you",
     * "baby", "chrome", "safari", "lint", "code", "body", "lint", "code" ], for
     * k = 3, return ["code", "lint", "baby"], for k = 4, return ["code",
     * "lint", "baby", "yes"].
     *
     * Challenge: Do it in O(nlogk) time and O(n) extra space. Extra points if
     * you can do it in O(n) time with O(k) extra space approximation
     * algorithms.
     *
     * @param words
     *            an array of string
     * @param k
     *            an integer
     * @return an array of string
     */
    @tags.HashTable
    @tags.PriorityQueue
    @tags.Heap
    public String[] topKFrequentWords(String[] words, int k) {
        Map<String, Integer> wordCount = new HashMap<>();
        for (String word : words) {
            if (wordCount.containsKey(word)) {
                wordCount.put(word, wordCount.get(word) + 1);
            } else {
                wordCount.put(word, 1);
            }
        }

        class WordCount {
            String word;
            int count;

            public WordCount(String word, int count) {
                this.word = word;
                this.count = count;
            }
        }

        PriorityQueue<WordCount> pq = new PriorityQueue<>(k,
                new Comparator<WordCount>() {
                    @Override
                    public int compare(WordCount wc1, WordCount wc2) {
                        if (wc1.count != wc2.count) {
                            return wc1.count - wc2.count;
                        } else {
                            return wc2.word.compareTo(wc1.word);
                        }
                    }
                });

        for (String word : wordCount.keySet()) {
            pq.offer(new WordCount(word, wordCount.get(word)));
            if (pq.size() > k) {
                pq.poll();
            }
        }

        String[] result = new String[k];
        for (int i = k - 1; i >= 0; i--) {
            result[i] = pq.poll().word;
        }

        return result;
    }

    /**
     * Wiggle Sort.
     *
     * Given an unsorted array nums, reorder it in-place such that nums[0] <=
     * nums[1] >= nums[2] <= nums[3]....
     *
     * Notice: Please complete the problem in-place.
     *
     * @param nums
     *            a list of integer
     * @return void
     */
    @tags.Array
    @tags.Sort
    @tags.QuickSort
    @tags.Company.Google
    public void wiggleSort(int[] nums) {
        if (nums != null) {
            Arrays.sort(nums);
            for (int i = 1; i + 1 < nums.length; i += 2) {
                int tmp = nums[i];
                nums[i] = nums[i + 1];
                nums[i + 1] = tmp;
            }
        }
    }

    /**
     * Wiggle Sort - better O(n) solution.
     */
    @tags.Array
    @tags.Sort
    @tags.QuickSort
    @tags.Company.Google
    public void wiggleSort2(int[] nums) {
        for (int i = 1; i < nums.length; i++) {
            if ((i % 2 == 1 && (nums[i] < nums[i - 1])
                    || (i % 2 == 0) && (nums[i] > nums[i - 1]))) {
                int temp = nums[i];
                nums[i] = nums[i - 1];
                nums[i - 1] = temp;
            }
        }
    }

    /**
     * Wiggle Sort II.
     *
     * Given an unsorted array nums, reorder it such that nums[0] < nums[1] >
     * nums[2] < nums[3]....
     *
     * Notice: You may assume all input has valid answer.
     *
     * Example: Given nums = [1, 5, 1, 1, 6, 4], one possible answer is [1, 4,
     * 1, 5, 1, 6]. Given nums = [1, 3, 2, 2, 3, 1], one possible answer is [2,
     * 3, 1, 3, 1, 2].
     *
     * Challenge: Can you do it in O(n) time and/or in-place with O(1) extra
     * space?
     *
     * This is O(nlogn) solution. O(n) time solution use partition to find
     * median, O(1) space requires index rewiring. Numbers in the middle are the
     * problem. They should be separated at the beginning and end of the array.
     *
     * @param nums
     *            a list of integer
     * @return void
     */
    @tags.Array
    @tags.QuickSort
    public void wiggleSortII(int[] nums) {
        Arrays.sort(nums);
        int[] temp = new int[nums.length];
        int s = (nums.length + 1) >> 1, t = nums.length;
        for (int i = 0; i < nums.length; i++) {
            temp[i] = (i & 1) == 0 ? nums[--s] : nums[--t];
        }

        for (int i = 0; i < nums.length; i++) {
            nums[i] = temp[i];
        }
    }

    // ---------------------------------------------------------------------- //
    // ------------------------------ Subarray ------------------------------ //
    // ---------------------------------------------------------------------- //

    /**
     * Subarray Sum.
     *
     * Given an integer array, find a subarray where the sum of numbers is zero.
     * Your code should return the index of the first number and the index of
     * the last number.
     *
     * Notice: There is at least one subarray that it's sum equals to zero.
     *
     * Example Given [-3, 1, 2, -3, 4], return [0, 2] or [1, 3].
     *
     * @param nums:
     *            A list of integers
     * @return: A list of integers includes the index of the first number and
     *          the index of the last number
     */
    @tags.Array
    @tags.HashTable
    @tags.Subarray
    public ArrayList<Integer> subarraySum(int[] nums) {
        ArrayList<Integer> result = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            return result;
        }

        // key = sum from start, value = current index
        Map<Integer, Integer> map = new HashMap<>();
        int sum = 0, index = -1;
        map.put(sum, index);

        // once same sum found again, subarray in the middle is the result
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            if (map.containsKey(sum)) {
                result.add(map.get(sum) + 1);
                result.add(i);
                return result;
            }
            map.put(sum, i);
        }

        return result;
    }

    /**
     * Subarray Sum II.
     *
     * Given an integer array, find a subarray where the sum of numbers is in a
     * given interval. Your code should return the number of possible answers.
     * (The element in the array should be positive)
     *
     * Example: Given [1,2,3,4] and interval = [1,3], return 4. The possible
     * answers are: [0, 0] [0, 1] [1, 1] [2, 2]
     *
     * This is O(n<sup>2</sup>) time solution. If all elements are positive,
     * then sum array will be increasing order, and time complexity will be
     * O(nlogn) with binary search to find the range.
     *
     * @param A
     *            an integer array
     * @param start
     *            an integer
     * @param end
     *            an integer
     * @return the number of possible answer
     */
    @tags.Array
    @tags.Subarray
    @tags.TwoPointers
    public int subarraySumII(int[] A, int start, int end) {
        int[] sums = new int[A.length + 1];
        sums[1] = A[0];
        for (int i = 2; i < A.length + 1; i++) {
            sums[i] = sums[i - 1] + A[i - 1];
        }

        int result = 0;
        for (int i = 0; i < A.length; i++) {
            for (int j = i + 1; j < A.length + 1; j++) {
                int sum = sums[j] - sums[i];
                if (sum >= start && sum <= end) {
                    result++;
                }
            }
        }

        return result;
    }

    /**
     * Subarray Sum Closest - O(nlogn) time.
     *
     * Given an integer array, find a subarray with sum closest to zero. Return
     * the indexes of the first number and last number.
     *
     * Example: Given [-3, 1, 1, -3, 5], return [0, 2], [1, 3], [1, 1], [2, 2]
     * or [0, 4].
     *
     * Challenge: O(nlogn) time.
     *
     * @param nums:
     *            A list of integers
     * @return: A list of integers includes the index of the first number and
     *          the index of the last number
     */
    @tags.Array
    @tags.Subarray
    @tags.Sort
    public int[] subarraySumClosest(int[] nums) {
        int[] range = new int[2];
        if (nums == null || nums.length == 0) {
            return range;
        }

        int[] sums = new int[nums.length + 1]; // prefix sum
        sums[0] = nums[0];
        Map<Integer, Integer> map = new HashMap<>(); // prefix sum to index map
        map.put(sums[0], 0);
        map.put(0, -1);

        // find the prefix sums for each index
        for (int i = 1; i < nums.length; i++) {
            sums[i] = sums[i - 1] + nums[i];

            // find the 0 sum
            if (map.containsKey(sums[i])) {
                range[0] = map.get(sums[i]) + 1;
                range[1] = i;
                return range;
            }

            map.put(sums[i], i);
        }

        Arrays.sort(sums);

        // compare each prefix sum and get the closest
        int minDiff = sums[1] - sums[0];
        range[0] = map.get(sums[1]);
        range[1] = map.get(sums[0]);
        for (int i = 2; i < sums.length; i++) {
            if (sums[i] - sums[i - 1] < minDiff) {
                minDiff = sums[i] - sums[i - 1];
                range[0] = map.get(sums[i]);
                range[1] = map.get(sums[i - 1]);
            }
        }

        Arrays.sort(range);
        range[0]++;
        return range;
    }

    /**
     * Minimum Size Subarray Sum.
     *
     * Given an array of n positive integers and a positive integer s, find the
     * minimal length of a subarray of which the sum ¡Ý s. If there isn't one,
     * return -1 instead.
     *
     * Example: Given the array [2,3,1,2,4,3] and s = 7, the subarray [4,3] has
     * the minimal length under the problem constraint.
     *
     * @param nums:
     *            an array of integers
     * @param s:
     *            an integer
     * @return: an integer representing the minimum size of subarray
     */
    @tags.Array
    @tags.Subarray
    @tags.TwoPointers
    @tags.Company.Facebook
    public int minimumSize(int[] nums, int s) {
        if (nums == null || nums.length == 0) {
            return -1;
        }

        int start = 0, end = 0, sum = nums[0], minSize = nums.length + 1;
        while (start <= end && end < nums.length) {
            if (sum >= s) {
                minSize = Math.min(end - start + 1, minSize);
                sum -= nums[start++];
            } else if (++end < nums.length) {
                sum += nums[end];
            }
        }
        return (minSize == nums.length + 1) ? -1 : minSize;
    }

    /**
     * Continuous Subarray Sum¡£
     *
     * Given an integer array, find a continuous subarray where the sum of
     * numbers is the biggest. Your code should return the index of the first
     * number and the index of the last number. (If their are duplicate answer,
     * return anyone).
     *
     * Example: Give [-3, 1, 3, -3, 4], return [1,4].
     *
     * @param A
     *            an integer array
     * @return A list of integers includes the index of the first number and the
     *         index of the last number
     */
    @tags.Array
    @tags.Subarray
    @tags.DynamicProgramming
    public ArrayList<Integer> continuousSubarraySum(int[] A) {
        ArrayList<Integer> range = new ArrayList<>();
        int max = A[0], maxEndHere = A[0];
        int start = 0, end = 0;
        int newStart = start, newEnd = end;
        for (int i = 1; i < A.length; i++) {
            if (maxEndHere > 0) {
                maxEndHere += A[i];
            } else {
                maxEndHere = A[i];
                newStart = i;
            }
            newEnd = i;

            if (maxEndHere > max) {
                max = maxEndHere;
                start = newStart;
                end = newEnd;
            }
        }

        range.add(start);
        range.add(end);
        return range;
    }

    /**
     * Continuous Subarray Sum II.
     *
     * Given an circular integer array (the next element of the last element is
     * the first element), find a continuous subarray in it, where the sum of
     * numbers is the biggest. Your code should return the index of the first
     * number and the index of the last number. If duplicate answers exist,
     * return any of them.
     *
     * Example: Give [3, 1, -100, -3, 4], return [4,1].
     *
     * @param A
     *            an integer array
     * @return A list of integers includes the index of the first number and the
     *         index of the last number
     */
    @tags.Array
    @tags.Subarray
    @tags.DynamicProgramming
    public ArrayList<Integer> continuousSubarraySumII(int[] A) {
        ArrayList<Integer> range = new ArrayList<>();
        range.add(0);
        range.add(0);
        int max = A[0], maxEndHere = A[0];
        int start = 0, end = 0;
        int total = A[0];

        for (int i = 1; i < A.length; i++) {
            // get total sum
            total += A[i];

            // get max sum
            if (maxEndHere > 0) {
                maxEndHere += A[i];
            } else {
                maxEndHere = A[i];
                start = i;
            }
            end = i;

            if (maxEndHere > max) {
                max = maxEndHere;
                range.set(0, start);
                range.set(1, end);
            }
        }

        int minEndHere = A[0];
        start = 0;
        end = 0;

        for (int i = 1; i < A.length; i++) {
            if (minEndHere < 0) {
                minEndHere += A[i];
            } else {
                minEndHere = A[i];
                start = i;
            }
            end = i;

            if (start == 0 && end == A.length - 1) {
                break;
            }

            if (total - minEndHere > max) {
                max = total - minEndHere;
                range.set(0, (end + 1) % A.length);
                range.set(1, (start - 1 + A.length) % A.length);
            }
        }

        return range;
    }

    /**
     * Maximum Subarray.
     *
     * Given an array of integers, find a contiguous subarray which has the
     * largest sum.
     *
     * Notice: The subarray should contain at least one number.
     *
     * Example: Given the array [-2,2,-3,4,-1,2,1,-5,3], the contiguous subarray
     * [4,-1,2,1] has the largest sum = 6.
     *
     * If you have figured out the O(n) solution, try coding another solution
     * using the divide and conquer approach, which is more subtle.
     *
     * @param nums:
     *            A list of integers
     * @return: A integer indicate the sum of max subarray
     */
    @tags.Array
    @tags.Subarray
    @tags.DynamicProgramming
    @tags.DivideAndConquer
    @tags.Greedy
    @tags.Enumeration
    @tags.Company.Bloomberg
    @tags.Company.LinkedIn
    @tags.Company.Microsoft
    @tags.Source.LintCode
    @tags.Status.OK
    public int maxSubArray(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }

        int max = nums[0];
        int maxEndHere = nums[0];
        for (int i = 1; i < nums.length; i++) {
            maxEndHere = (maxEndHere <= 0) ? nums[i] : maxEndHere + nums[i];
            if (maxEndHere > max) {
                max = maxEndHere;
            }
        }

        return max;
    }

    /**
     * Maximum Subarray II.
     *
     * Given an array of integers, find two non-overlapping subarrays which have
     * the largest sum. The number in each subarray should be contiguous. Return
     * the largest sum.
     *
     * Notice: The subarray should contain at least one number
     *
     * Example: For given [1, 3, -1, 2, -1, 2], the two subarrays are [1, 3] and
     * [2, -1, 2] or [1, 3, -1, 2] and [2], they both have the largest sum 7.
     *
     * Challenge: Can you do it in time complexity O(n) ?
     *
     * @param nums:
     *            A list of integers
     * @return: An integer denotes the sum of max two non-overlapping subarrays
     */
    @tags.Array
    @tags.Subarray
    @tags.Greedy
    @tags.Enumeration
    @tags.ForwardBackwardTraversal
    @tags.Source.LintCode
    @tags.Status.NeedPractice
    public int maxTwoSubArrays(ArrayList<Integer> nums) {
        if (nums == null || nums.size() < 2) {
            return 0;
        }

        int len = nums.size();
        int[] forwardMax = new int[len];
        int maxToHere = nums.get(0);
        forwardMax[0] = nums.get(0);
        for (int i = 1; i < len; i++) {
            maxToHere = Math.max(nums.get(i), maxToHere + nums.get(i));
            forwardMax[i] = Math.max(forwardMax[i - 1], maxToHere);
        }

        int[] backwardMax = new int[len];
        maxToHere = nums.get(len - 1);
        backwardMax[len - 1] = nums.get(len - 1);
        for (int i = len - 2; i >= 0; i--) {
            maxToHere = Math.max(nums.get(i), maxToHere + nums.get(i));
            backwardMax[i] = Math.max(backwardMax[i + 1], maxToHere);
        }

        int max = forwardMax[0] + backwardMax[1];
        for (int i = 1; i < len - 1; i++) {
            max = Math.max(max, forwardMax[i] + backwardMax[i + 1]);
        }

        return max;
    }

    /**
     * Maximum Subarray III.
     *
     * Given an array of integers and a number k, find k non-overlapping
     * subarrays which have the largest sum. The number in each subarray should
     * be contiguous. Return the largest sum.
     *
     * Notice: The subarray should contain at least one number
     *
     * Example: Given [-1,4,-2,3,-2,3], k=2, return 8
     *
     * @param nums:
     *            A list of integers
     * @param k:
     *            An integer denote to find k non-overlapping subarrays
     * @return: An integer denote the sum of max k non-overlapping subarrays
     */
    @tags.Array
    @tags.Subarray
    @tags.DynamicProgramming
    @tags.Source.LintCode
    @tags.Status.Hard
    public int maxSubArray(int[] nums, int k) {
        // may need to discuss about the optimization

        int n = nums.length;
        int[][] maxToHere = new int[n + 1][k + 1];
        int[][] maxGlobal = new int[n + 1][k + 1];

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= i && j <= k; j++) {
                if (i == j) {
                    maxToHere[i][j] = maxGlobal[i - 1][j - 1] + nums[i - 1];
                    maxGlobal[i][j] = maxToHere[i][j];
                } else {
                    maxToHere[i][j] = Math.max(maxGlobal[i - 1][j - 1],
                            maxToHere[i - 1][j]) + nums[i - 1];
                    maxGlobal[i][j] = Math.max(maxToHere[i][j],
                            maxGlobal[i - 1][j]);
                }
            }
        }
        return maxGlobal[n][k];
    }

    /**
     * Maximum Subarray Difference.
     *
     * Given an array with integers. Find two non-overlapping subarrays A and B,
     * which |SUM(A) - SUM(B)| is the largest. Return the largest difference.
     *
     * Notice: The subarray should contain at least one number.
     *
     * Example: For [1, 2, -3, 1], return 6.
     *
     * Challenge: O(n) time and O(n) space.
     *
     * @param nums:
     *            A list of integers
     * @return: An integer indicate the value of maximum difference between two
     *          Subarrays
     */
    @tags.Array
    @tags.Subarray
    @tags.Greedy
    @tags.Enumeration
    @tags.ForwardBackwardTraversal
    @tags.DynamicProgramming
    @tags.Source.LintCode
    @tags.Status.NeedPractice
    public int maxDiffSubArrays(int[] nums) {
        if (nums == null || nums.length < 2) {
            return 0;
        }

        int n = nums.length;
        int[] forwardMax = new int[n], forwardMin = new int[n];
        forwardMax[0] = nums[0];
        forwardMin[0] = nums[0];
        int maxEndHere = nums[0], minEndHere = nums[0];

        for (int i = 1; i < n; i++) {
            maxEndHere = maxEndHere <= 0 ? nums[i] : maxEndHere + nums[i];
            forwardMax[i] = Math.max(forwardMax[i - 1], maxEndHere);
            minEndHere = minEndHere >= 0 ? nums[i] : minEndHere + nums[i];
            forwardMin[i] = Math.min(forwardMin[i - 1], minEndHere);
        }

        int[] backwardMax = new int[n], backwardMin = new int[n];
        backwardMax[n - 1] = nums[n - 1];
        backwardMin[n - 1] = nums[n - 1];
        int maxStartHere = nums[n - 1], minStartHere = nums[n - 1];

        for (int i = n - 2; i >= 0; i--) {
            maxStartHere = maxStartHere <= 0 ? nums[i] : maxStartHere + nums[i];
            backwardMax[i] = Math.max(backwardMax[i + 1], maxStartHere);
            minStartHere = minStartHere >= 0 ? nums[i] : minStartHere + nums[i];
            backwardMin[i] = Math.min(backwardMin[i + 1], minStartHere);
        }

        int max = 0;
        for (int i = 0; i < n - 1; i++) {
            max = Math.max(max, forwardMax[i] - backwardMin[i + 1]);
            max = Math.max(max, backwardMax[i + 1] - forwardMin[i]);
        }
        return max;
    }

    /**
     * Minimum Subarray
     *
     * Given an array of integers, find the subarray with smallest sum. Return
     * the sum of the subarray.
     *
     * The subarray should contain one integer at least.
     *
     * Example For [1, -1, -2, 1], return -3.
     *
     * @param nums:
     *            a list of integers
     * @return: A integer indicate the sum of minimum subarray
     */
    @tags.Array
    @tags.Subarray
    @tags.DynamicProgramming
    @tags.Greedy
    @tags.Source.LintCode
    @tags.Status.Easy
    public int minSubArray(ArrayList<Integer> nums) {
        if (nums == null || nums.size() == 0) {
            return 0;
        }

        int min = nums.get(0);
        int minEndHere = nums.get(0);

        for (int i = 1; i < nums.size(); i++) {
            if (minEndHere > 0) {
                minEndHere = 0;
            }
            minEndHere += nums.get(i);
            min = Math.min(min, minEndHere);
        }

        return min;
    }

    /**
     * Maximum Product Subarray.
     *
     * Find the contiguous subarray within an array (containing at least one
     * number) which has the largest product.
     *
     * Example: For example, given the array [2,3,-2,4], the contiguous subarray
     * [2,3] has the largest product = 6.
     *
     * @param nums:
     *            an array of integers
     * @return: an integer
     */
    @tags.Array
    @tags.Subarray
    @tags.DynamicProgramming
    @tags.Company.LinkedIn
    @tags.Status.OK
    public int maxProduct(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }

        int max = nums[0];
        int maxToHere = nums[0];
        int minToHere = nums[0];

        for (int i = 1; i < nums.length; i++) {
            int[] minMax = new int[3];
            minMax[0] = nums[i];
            minMax[1] = maxToHere * nums[i];
            minMax[2] = minToHere * nums[i];
            Arrays.sort(minMax);
            maxToHere = minMax[2];
            minToHere = minMax[0];
            max = Math.max(max, maxToHere);
        }

        return max;
    }

    /**
     * Maximum Size Subarray Sum Equals k.
     *
     * Given an array nums and a target value k, find the maximum length of a
     * subarray that sums to k. If there isn't one, return 0 instead.
     *
     * Note: The sum of the entire nums array is guaranteed to fit within the
     * 32-bit signed integer range.
     *
     * Example 1: Given nums = [1, -1, 5, -2, 3], k = 3, return 4. (because the
     * subarray [1, -1, 5, -2] sums to 3 and is the longest)
     *
     * Example 2: Given nums = [-2, -1, 2, 1], k = 1, return 2. (because the
     * subarray [-1, 2] sums to 1 and is the longest)
     *
     * Follow Up: Can you do it in O(n) time?
     *
     * @param nums
     * @param k
     * @return
     */
    @tags.HashTable
    @tags.Company.Facebook
    @tags.Company.Palantir
    @tags.Status.Hard
    public int maxSubArrayLen(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return 0;
        }

        int n = nums.length;
        int[] sums = new int[n + 1];

        // get prefix sum array
        for (int i = n - 1; i >= 0; i--) {
            sums[i] = sums[i + 1] + nums[i];
        }

        Map<Integer, Integer> map = new HashMap<>();
        int maxLen = 0;

        for (int i = n; i >= 0; i--) {
            int x = sums[i] - k;
            if (map.containsKey(x)) {
                int len = map.get(x) - i;
                if (maxLen < len) {
                    maxLen = len;
                }
            }
            if (!map.containsKey(sums[i])) {
                map.put(sums[i], i);
            }
        }

        return maxLen;
    }

    // ---------------------------------------------------------------------- //
    // ------------------------------ Two Sum ------------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Two Sum - O(n).
     *
     * Given an array of integers, find two numbers such that they add up to a
     * specific target number.
     *
     * The function twoSum should return indices of the two numbers such that
     * they add up to the target, where index1 must be less than index2. Please
     * note that your returned answers (both index1 and index2) are not
     * zero-based.
     *
     * You may assume that each input would have exactly one solution.
     *
     * Example: Input: numbers={2, 7, 11, 15}, target=9 Output: [1, 2].
     *
     * @param numbers
     *            An array of Integer
     * @param target
     *            target = numbers[index1] + numbers[index2]
     * @return : [index1 + 1, index2 + 1] (index1 < index2)
     */
    @tags.Array
    @tags.TwoPointers
    @tags.Sort
    @tags.HashTable
    @tags.Company.Adobe
    @tags.Company.Airbnb
    @tags.Company.Amazon
    @tags.Company.Apple
    @tags.Company.Bloomberg
    @tags.Company.Dropbox
    @tags.Company.Facebook
    @tags.Company.LinkedIn
    @tags.Company.Microsoft
    @tags.Company.Uber
    @tags.Company.Yahoo
    @tags.Company.Yelp
    @tags.Status.OK
    public int[] twoSum(int[] numbers, int target) {
        // use map instead of 2 pointers since sorting is nlogn

        int[] result = new int[2];
        if (numbers == null || numbers.length < 2) {
            return result;
        }

        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < numbers.length; i++) {
            int theOther = target - numbers[i];
            if (map.containsKey(theOther)) {
                result[0] = map.get(theOther);
                result[1] = i + 1;
                return result;
            }
            map.put(numbers[i], i + 1);
        }
        return result;
    }

    /**
     * Two Sum II.
     *
     * Given an array of integers, find how many pairs in the array such that
     * their sum is bigger than a specific target number. Please return the
     * number of pairs.
     *
     * Example: Given numbers = [2, 7, 11, 15], target = 24. Return 1. (11 + 15
     * is the only pair)
     *
     * Challenge: Do it in O(1) extra space and O(nlogn) time.
     *
     * @param nums:
     *            an array of integer
     * @param target:
     *            an integer
     * @return: an integer
     */
    @tags.Array
    @tags.TwoPointers
    @tags.Sort
    public int twoSum2(int[] nums, int target) {
        if (nums == null || nums.length < 2) {
            return 0;
        }

        Arrays.sort(nums);
        int result = 0;
        for (int start = 0, end = nums.length - 1; start < end;) {
            int twoSum = nums[start] + nums[end];
            if (twoSum <= target) {
                start++;
            } else {
                result += (end - start);
                end--;
            }
        }

        return result;
    }

    /**
     * Two Sum III - Data structure design.
     *
     * Design and implement a TwoSum class. It should support the following
     * operations: add and find.
     *
     * add - Add the number to an internal data structure. find - Find if there
     * exists any pair of numbers which sum is equal to the value.
     *
     * For example, add(1); add(3); add(5); find(4) -> true. find(7) -> false.
     */
    @tags.Design
    @tags.HashTable
    @tags.Company.LinkedIn
    public class TwoSum {
        // Your TwoSum object will be instantiated and called as such:
        // TwoSum twoSum = new TwoSum();
        // twoSum.add(number);
        // twoSum.find(value);

        Map<Integer, Integer> nums = new HashMap<>();

        // Add the number to an internal data structure.
        public void add(int number) {
            if (nums.containsKey(number)) {
                nums.put(number, nums.get(number) + 1);
            } else {
                nums.put(number, 1);
            }
        }

        // Find if there exists any pair of numbers which sum is equal to the
        // value.
        public boolean find(int value) {
            for (Integer first : nums.keySet()) {
                int second = value - first;
                if (nums.containsKey(second)) {
                    if (first == second && nums.get(second) == 1) {
                        continue;
                    }
                    return true;
                }
            }
            return false;
        }
    }

    /**
     * Two Sum Closest.
     *
     * Given an array nums of n integers, find two integers in nums such that
     * the sum is closest to a given number, target. Return the difference
     * between the sum of the two integers and the target.
     *
     * Example: Given array nums = [-1, 2, 1, -4], and target = 4. The minimum
     * difference is 1. (4 - (2 + 1) = 1).
     *
     * @param nums
     *            an integer array
     * @param target
     *            an integer
     * @return the difference between the sum and the target
     */
    @tags.TwoPointers
    @tags.Sort
    public int twoSumCloset(int[] nums, int target) {
        if (nums == null || nums.length < 2) {
            return -1;
        }

        Arrays.sort(nums);

        int minDiff = Integer.MAX_VALUE;
        for (int i = 0, j = nums.length - 1; i < j;) {
            int diff = nums[i] + nums[j] - target;
            if (diff == 0) {
                return 0;
            } else {
                if (diff > 0) {
                    j--;
                } else {
                    i++;
                }
                minDiff = Math.min(Math.abs(diff), minDiff);
            }
        }

        return minDiff;
    }

    /**
     * 3Sum.
     *
     * Given an array S of n integers, are there elements a, b, c in S such that
     * a + b + c = 0? Find all unique triplets in the array which gives the sum
     * of zero.
     *
     * Note: Elements in a triplet (a,b,c) must be in non-descending order. (ie,
     * a ¡Ü b ¡Ü c) The solution set must not contain duplicate triplets.
     *
     * For example, given array S = {-1 0 1 2 -1 -4}, A solution set is: (-1, 0,
     * 1) (-1, -1, 2)
     *
     * @param numbers
     *            : Give an array numbers of n integer
     * @return : Find all unique triplets in the array which gives the sum of
     *         zero.
     */
    @tags.Array
    @tags.Sort
    @tags.TwoPointers
    @tags.Company.Adobe
    @tags.Company.Amazon
    @tags.Company.Bloomberg
    @tags.Company.Facebook
    @tags.Company.Microsoft
    public ArrayList<ArrayList<Integer>> threeSum(int[] numbers) {
        Arrays.sort(numbers);
        int n = numbers.length;
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        
        for (int i = 0; i < n - 2; i++) {
            if (i != 0 && numbers[i] == numbers[i - 1]) {
                continue;
            }

            int target = -numbers[i];
            int start = i + 1, end = n - 1;

            while (start < end) {
                int twoSum = numbers[start] + numbers[end];
                if (twoSum > target) {
                    end--;
                } else if (twoSum < target) {
                    start++;
                } else {
                    ArrayList<Integer> three = new ArrayList<>();
                    three.add(numbers[i]);
                    three.add(numbers[start++]);
                    three.add(numbers[end--]);
                    result.add(three);
                    
                    while (start < end && numbers[start] == numbers[start - 1]) {
                        start++;
                    }
                    while (start < end && numbers[end] == numbers[end + 1]) {
                        end--;
                    }
                }
            }
        }
        
        return result;
    }

    /**
     * 3Sum Smaller.
     *
     * Given an array of n integers nums and a target, find the number of index
     * triplets i, j, k with 0 <= i < j < k < n that satisfy the condition
     * nums[i] + nums[j] + nums[k] < target.
     *
     * For example, given nums = [-2, 0, 1, 3], and target = 2. Return 2.
     * Because there are two triplets which sums are less than 2: [-2, 0, 1]
     * [-2, 0, 3].
     *
     * Follow up: Could you solve it in O(n<sup>2</sup>) runtime?
     *
     * @param nums
     * @param target
     * @return
     */
    @tags.Array
    @tags.TwoPointers
    @tags.Company.Google
    @tags.Status.NeedPractice
    public int threeSumSmaller(int[] nums, int target) {
        Arrays.sort(nums);
        int count = 0;

        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1, k = nums.length - 1; j < k;) {
                if (nums[i] + nums[j] + nums[k] >= target) {
                    k--;
                } else {
                    count += k - j;
                    j++;
                }
            }
        }

        return count;
    }

    /**
     * 3Sum Closest
     *
     * Given an array S of n integers, find three integers in S such that the
     * sum is closest to a given number, target. Return the sum of the three
     * integers. You may assume that each input would have exactly one solution.
     *
     * For example, given array S = {-1 2 1 -4}, and target = 1. The sum that is
     * closest to the target is 2. (-1 + 2 + 1 = 2).
     *
     * @param numbers:
     *            Give an array numbers of n integer
     * @param target
     *            : An integer
     * @return : return the sum of the three integers, the sum closest target.
     */
    @tags.Array
    @tags.Sort
    @tags.TwoPointers
    @tags.Company.Bloomberg
    public int threeSumClosest(int[] num, int target) {
        Arrays.sort(num);
        int sum = num[0] + num[1] + num[2];

        // traverse the array for every possible position of first number
        for (int i = 0; i < num.length - 2; i++) {
            // second number start from the one next to first one
            // third number start from the last number in the array
            for (int j = i + 1, k = num.length - 1; j < k;) {
                int threeSum = num[i] + num[j] + num[k];

                // compare temp with target
                if (threeSum == target) {
                    return threeSum;
                } else {
                    // update sum
                    if (Math.abs(threeSum - target) < Math.abs(sum - target)) {
                        sum = threeSum;
                    }

                    // update j and k
                    if (threeSum > target) {
                        k--;
                    } else {
                        j++;
                    }
                }
            }
        }

        return sum;
    }

    /**
     * Triangle Count.
     *
     * Given an array of integers, how many three numbers can be found in the
     * array, so that we can build an triangle whose three edges length is the
     * three numbers that we find?
     *
     * Example: Given array S = [3,4,6,7], return 3. They are: [3,4,6] [3,6,7]
     * [4,6,7]. Given array S = [4,4,4,4], return 4. They are: [4(1),4(2),4(3)]
     * [4(1),4(2),4(4)] [4(1),4(3),4(4)] [4(2),4(3),4(4)].
     *
     * @param S:
     *            A list of integers
     * @return: An integer
     */
    @tags.Array
    @tags.TwoPointers
    @tags.Source.LintCode
    public int triangleCount(int S[]) {
        Arrays.sort(S);

        int result = 0;
        for (int i = 2; i < S.length; i++) {
            for (int j = 0, k = i - 1; j < k;) {
                if (S[j] + S[k] > S[i]) {
                    result += k - j;
                    k--;
                } else {
                    j++;
                }
            }
        }

        return result;
    }

    /**
     * 4Sum - O(n<sup>3</sup>) time.
     *
     * Given an array S of n integers, are there elements a, b, c, and d in S
     * such that a + b + c + d = target? Find all unique quadruplets in the
     * array which gives the sum of target.
     *
     * Note: Elements in a quadruplet (a,b,c,d) must be in non-descending order.
     * (ie, a ¡Ü b ¡Ü c ¡Ü d) The solution set must not contain duplicate
     * quadruplets.
     *
     * For example, given array S = {1 0 -1 0 -2 2}, and target = 0. A solution
     * set is: (-1, 0, 0, 1) (-2, -1, 1, 2) (-2, 0, 0, 2)
     *
     * @param numbers
     *            : Give an array numbersbers of n integer
     * @param target
     *            : you need to find four elements that's sum of target
     * @return : Find all unique quadruplets in the array which gives the sum of
     *         zero.
     */
    @tags.Array
    @tags.Sort
    @tags.TwoPointers
    @tags.HashTable
    public ArrayList<ArrayList<Integer>> fourSum(int[] numbers, int target) {
        Arrays.sort(numbers);
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        int n = numbers.length;

        for (int i = 0; i < n - 3; i++) {
            if (i != 0 && numbers[i] == numbers[i - 1]) {
                continue;
            }

            for (int j = i + 1; j < n - 2; j++) {
                if (j != i + 1 && numbers[j] == numbers[j - 1]) {
                    continue;
                }

                int start = j + 1, end = n - 1;
                while (start < end) {
                    int sum = numbers[i] + numbers[j] + numbers[start]
                            + numbers[end];
                    if (sum < target) {
                        start++;
                    } else if (sum > target) {
                        end--;
                    } else {
                        ArrayList<Integer> group = new ArrayList<>();
                        group.add(numbers[i]);
                        group.add(numbers[j]);
                        group.add(numbers[start++]);
                        group.add(numbers[end--]);
                        result.add(group);

                        while (start < end
                                && numbers[start] == numbers[start - 1]) {
                            start++;
                        }
                        while (start < end
                                && numbers[end] == numbers[end + 1]) {
                            end--;
                        }
                    }
                }
            }
        }

        return result;
    }

    /**
     * 4Sum.
     *
     * O(n<sup>3</sup>) or O(n<sup>2</sup>logn<sup>2</sup>) =
     * O(n<sup>2</sup>logn) time.
     */
    @tags.Array
    @tags.Sort
    @tags.TwoPointers
    @tags.HashTable
    public ArrayList<ArrayList<Integer>> fourSum2(int[] num, int target) {
        int len = num.length;
        Map<Integer, ArrayList<ArrayList<Integer>>> map = new HashMap<Integer, ArrayList<ArrayList<Integer>>>();
        Set<ArrayList<Integer>> set = new HashSet<ArrayList<Integer>>();

        for (int i = 0; i < len; i++) {
            for (int j = i + 1; j < len; j++) {
                int sum = num[i] + num[j];
                ArrayList<Integer> two = new ArrayList<Integer>();
                two.add(i);
                two.add(j);

                if (map.containsKey(sum)) {
                    map.get(sum).add(two);
                } else {
                    ArrayList<ArrayList<Integer>> list = new ArrayList<ArrayList<Integer>>();
                    list.add(two);
                    map.put(sum, list);
                }
            }
        }

        for (int i = 0; i < len; i++) {
            for (int j = i + 1; j < len; j++) {
                int sum = num[i] + num[j];
                int target2 = target - sum;

                if (map.containsKey(target2)) {
                    ArrayList<ArrayList<Integer>> two = map.get(target2);
                    for (ArrayList<Integer> list : two) {
                        int x = list.get(0);
                        int y = list.get(1);
                        if (x == i || x == j || y == i || y == j)
                            break;
                        int[] temp = new int[4];
                        temp[0] = num[i];
                        temp[1] = num[j];
                        temp[2] = num[x];
                        temp[3] = num[y];
                        Arrays.sort(temp);

                        ArrayList<Integer> four = new ArrayList<Integer>();
                        for (int z : temp)
                            four.add(z);

                        set.add(four);
                    }
                }
            }
        }

        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>(
                set);
        return result;
    }

    /**
     * k Sum
     *
     * Given n distinct positive integers, integer k (k <= n) and a number
     * target. Find k numbers where sum is target. Calculate how many solutions
     * there are?
     *
     * Example Given [1,2,3,4], k = 2, target = 5. There are 2 solutions: [1,4]
     * and [2,3]. Return 2.
     *
     * @param A:
     *            an integer array.
     * @param k:
     *            a positive integer (k <= length(A))
     * @param target:
     *            a integer
     * @return an integer
     */
    @tags.DynamicProgramming
    @tags.Source.LintCode
    public int kSum(int A[], int k, int target) {
        if (A == null || A.length == 0 || k == 0 || target == 0) {
            return 0;
        }

        int[][][] dp = new int[A.length + 1][k + 1][target + 1];
        for (int i = 0; i <= A.length; i++) {
            dp[i][0][0] = 1;
        }

        for (int i = 1; i <= A.length; i++) {
            for (int j = 1; j <= k; j++) {
                for (int t = 1; t <= target; t++) {
                    dp[i][j][t] = dp[i - 1][j][t];
                    if (A[i - 1] <= t) {
                        dp[i][j][t] += dp[i - 1][j - 1][t - A[i - 1]];
                    }
                }
            }
        }

        return dp[A.length][k][target];
    }

    /**
     * k Sum II.
     *
     * Given n unique integers, number k (1<=k<=n) and target. Find all possible
     * k integers where their sum is target.
     *
     * Example: Given [1,2,3,4], k = 2, target = 5. Return: [ [1,4], [2,3] ].
     *
     * @param A:
     *            an integer array.
     * @param k:
     *            a positive integer (k <= length(A))
     * @param target:
     *            a integer
     * @return a list of lists of integer
     */
    @tags.DFS
    @tags.Backtracking
    @tags.Source.LintCode
    public ArrayList<ArrayList<Integer>> kSumII(int[] A, int k, int target) {
        if (A == null || A.length == 0) {
            return new ArrayList<>();
        }

        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        ArrayList<Integer> path = new ArrayList<>();
        kSumII(A, 0, k, target, result, path);
        return result;
    }

    private void kSumII(int[] A, int pos, int k, int target,
            ArrayList<ArrayList<Integer>> result, ArrayList<Integer> path) {
        if (target == 0 && k == 0) {
            result.add(new ArrayList<>(path));
            return;
        } else if (k == 0 || pos == A.length) {
            return;
        }

        path.add(A[pos]);
        kSumII(A, pos + 1, k - 1, target - A[pos], result, path);
        path.remove(path.size() - 1);
        kSumII(A, pos + 1, k, target, result, path);
    }

    // ---------------------------------------------------------------------- //
    // ------------------ Best Time to Buy and Sell Stock ------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Best Time to Buy and Sell Stock
     *
     * Say you have an array for which the ith element is the price of a given
     * stock on day i. If you were only permitted to complete at most one
     * transaction (ie, buy one and sell one share of the stock), design an
     * algorithm to find the maximum profit.
     *
     * Given array [3,2,3,1,2], return 1.
     *
     * @param prices:
     *            Given an integer array
     * @return: Maximum profit
     */
    @tags.Array
    @tags.DynamicProgramming
    @tags.Greedy
    @tags.Enumeration
    @tags.Company.Facebook
    @tags.Company.Uber
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }

        int min = prices[0];
        int profit = 0;

        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > min) {
                profit = Math.max(profit, prices[i] - min);
            } else if (prices[i] < min) {
                min = prices[i];
            }
        }

        return profit;
    }

    /**
     * Best Time to Buy and Sell Stock II
     *
     * Say you have an array for which the ith element is the price of a given
     * stock on day i. Design an algorithm to find the maximum profit. You may
     * complete as many transactions as you like (ie, buy one and sell one share
     * of the stock multiple times). However, you may not engage in multiple
     * transactions at the same time (ie, you must sell the stock before you buy
     * again).
     *
     * Example: Given an example [2,1,2,0,1], return 2.
     *
     * @param prices:
     *            Given an integer array
     * @return: Maximum profit
     */
    @tags.Array
    @tags.Greedy
    @tags.Enumeration
    public int maxProfitII(int[] prices) {
        if (prices == null || prices.length < 2) {
            return 0;
        }

        int profit = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1]) {
                profit += (prices[i] - prices[i - 1]);
            }
        }

        return profit;
    }

    /**
     * Best Time to Buy and Sell Stock III
     *
     * Say you have an array for which the ith element is the price of a given
     * stock on day i. Design an algorithm to find the maximum profit. You may
     * complete at most two transactions.
     *
     * Note: You may not engage in multiple transactions at the same time (ie,
     * you must sell the stock before you buy again).
     *
     * Example: Given an example [4,4,6,1,1,4,2,5], return 6.
     *
     * @param prices:
     *            Given an integer array
     * @return: Maximum profit
     */
    @tags.Array
    @tags.ForwardBackwardTraversal
    @tags.DynamicProgramming
    @tags.Enumeration
    public int maxProfitIII(int[] prices) {
        if (prices == null || prices.length < 2) {
            return 0;
        }

        int len = prices.length;
        int[] forward = new int[len];
        int min = prices[0];
        for (int i = 1; i < len; i++) {
            forward[i] = Math.max(forward[i - 1], prices[i] - min);
            min = Math.min(min, prices[i]);
        }

        int[] backward = new int[len];
        int max = prices[len - 1];
        for (int i = len - 2; i >= 0; i--) {
            backward[i] = Math.max(backward[i + 1], max - prices[i]);
            max = Math.max(max, prices[i]);
        }

        int maxProfit = 0;
        for (int i = 0; i < len; i++) {
            maxProfit = Math.max(maxProfit, forward[i] + backward[i]);
        }

        return maxProfit;
    }

    /**
     * Best Time to Buy and Sell Stock IV
     *
     * Say you have an array for which the ith element is the price of a given
     * stock on day i. Design an algorithm to find the maximum profit. You may
     * complete at most k transactions.
     *
     * Notice: You may not engage in multiple transactions at the same time
     * (i.e., you must sell the stock before you buy again).
     *
     * Example: Given prices = [4,4,6,1,1,4,2,5], and k = 2, return 6.
     *
     * @param k:
     *            An integer
     * @param prices:
     *            Given an integer array
     * @return: Maximum profit
     */
    @tags.Array
    @tags.DynamicProgramming
    public int maxProfitIV(int k, int[] prices) {
        if (k == 0) {
            return 0;
        }
        if (k >= prices.length / 2) {
            int profit = 0;
            for (int i = 1; i < prices.length; i++) {
                if (prices[i] > prices[i - 1]) {
                    profit += prices[i] - prices[i - 1];
                }
            }
            return profit;
        }
        int len = prices.length;
        int[][] mustsell = new int[len][k + 1];
        int[][] globalbest = new int[len][k + 1];

        mustsell[0][0] = globalbest[0][0] = 0;
        for (int i = 1; i <= k; i++) {
            mustsell[0][i] = globalbest[0][i] = 0;
        }

        for (int i = 1; i < len; i++) {
            int gainorlose = prices[i] - prices[i - 1];
            mustsell[i][0] = 0;
            for (int j = 1; j <= k; j++) {
                mustsell[i][j] = Math.max(
                        globalbest[(i - 1)][j - 1] + gainorlose,
                        mustsell[(i - 1)][j] + gainorlose);
                globalbest[i][j] = Math.max(globalbest[(i - 1)][j],
                        mustsell[i][j]);
            }
        }
        return globalbest[(len - 1)][k];
    }

    /**
     * Best Time to Buy and Sell Stock with Cooldown.
     *
     * Say you have an array for which the ith element is the price of a given
     * stock on day i. Design an algorithm to find the maximum profit. You may
     * complete as many transactions as you like (ie, buy one and sell one share
     * of the stock multiple times) with the following restrictions:
     *
     * You may not engage in multiple transactions at the same time (ie, you
     * must sell the stock before you buy again). After you sell your stock, you
     * cannot buy stock on next day. (ie, cooldown 1 day)
     *
     * Example: prices = [1, 2, 3, 0, 2] maxProfit = 3 transactions = [buy,
     * sell, cooldown, buy, sell].
     *
     * @param prices:
     *            Given an integer array
     * @return Maximum profit
     */
    @tags.Array
    @tags.DynamicProgramming
    @tags.Source.LeetCode
    public int maxProfitWithCooldown(int[] prices) {
        if (prices == null || prices.length < 2) {
            return 0;
        }

        int len = prices.length;
        int[] local = new int[len + 1];
        int[] global = new int[len + 1];
        local[2] = prices[1] - prices[0];
        global[2] = Math.max(0, local[2]);

        for (int i = 3; i <= len; i++) {
            int loseOrGain = prices[i - 1] - prices[i - 2];
            local[i] = Math.max(global[i - 3], local[i - 1]) + loseOrGain;
            global[i] = Math.max(local[i], global[i - 1]);
        }
        return global[len];
    }

    // ---------------------------------- OLD ----------------------------------

    /**
     * Find Duplicate from Unsorted Array
     * 
     * A = [2, 3, 4, 2, 5], length = n, containing one missing, one duplicate
     * 
     * @param A
     * @return new length
     */
    public int findDup(int[] num) {
        if (num == null || num.length == 0) {
            return -1;
        }

        for (int i = 0; i < num.length; i++) {
            if (num[num[i] - 1] == num[i]) {
                return num[i];
            } else {
                if (num[i] == i + 1)
                    continue;
                else {
                    int j = num[i] - 1;
                    num[i] = num[j];
                    num[j] = j + 1;
                    i--;
                }
            }
        }

        return -1;
    }

    public int subArrayWithValue(int[] A) {
        return 0;
    }

    public int subArrayCloseToValue(int[] A) {
        return 0;
    }

    public int maxSubMatrix(int[][] A) {
        if (A == null || A.length == 0 || A[0].length == 0) {
            return 0;
        }

        int max = A[0][0];

        for (int i = 0; i < A.length; i++) {
            for (int j = i; j < A.length; j++) {

            }
        }
        return 0;
    }

    /**
     * Longest Substring Without Repeating Characters
     *
     * Given a string, find the length of the longest substring without
     * repeating characters. For example, the longest substring without
     * repeating letters for "abcabcbb" is "abc", which the length is 3. For
     * "bbbbb" the longest substring is "b", with the length of 1.
     */
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }

        int start = 0;
        int max = 1;
        Set<Character> set = new HashSet<Character>();

        for (int i = 0; i < s.length(); i++) {
            while (set.contains(s.charAt(i))) {
                set.remove(s.charAt(start++));
            }
            set.add(s.charAt(i));
            max = Math.max(max, set.size());
        }

        return max;
    }

    // ---------------------------------------------------------------------- //
    // ------------------------------ Unit Tests ---------------------------- //
    // ---------------------------------------------------------------------- //

    @Test
    public void test() {
        findDupTest();
        topkTest();
        subarraySumClosestTest();
        continuousSubarraySumTest();
        continuousSubarraySumIITest();
    }

    private void findDupTest() {
        int[] findDup = { 2, 3, 4, 2, 5 };
        System.out.println(findDup(findDup));
    }

    private void topkTest() {
        int[] nums = { 3, 10, 1000, -99, 4, 100 };
        int k = 3;
        int[] result = topk(nums, k);
        Assert.assertEquals(1000, result[0]);
        Assert.assertEquals(100, result[1]);
        Assert.assertEquals(10, result[2]);
    }

    private void subarraySumClosestTest() {
        int[] nums = {-3,1,1,-3,5};
        int[] result = subarraySumClosest(nums);
        Assert.assertEquals(1, result[0]);
        Assert.assertEquals(3, result[1]);
    }

    private void continuousSubarraySumTest() {
        int[] nums = { 1, 1, 1, 1, 1, 1, 1, 1, 1, -19, 1, 1, 1, 1, 1, 1, 1, -2,
                1, 1, 1, 1, 1, 1, 1, 1, -2, 1, -15, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1 };
        continuousSubarraySum(nums);
    }

    private void continuousSubarraySumIITest() {
        int[] nums = { 2, -1, -2, -3, -100, 1, 2, 3, 100 };
        ArrayList<Integer> range = continuousSubarraySumII(nums);
        Assert.assertTrue(range.get(0) == 5 && range.get(1) == 0);

        int[] nums2 = { 29, 84, -44, 17, -22, 40, -5, 19, 90 };
        range = continuousSubarraySumII(nums2);
        Assert.assertTrue(range.get(0) == 5 && range.get(1) == 1);

        int[] nums3 = { -5, 10, 5, -3, 1, 1, 1, -2, 3, -4 };
        range = continuousSubarraySumII(nums3);
        Assert.assertTrue(range.get(0) == 1 && range.get(1) == 8);
    }

}
