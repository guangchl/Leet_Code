package categories;

import java.util.ArrayList;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

public class BinarySearch {

    // ---------------------------------------------------------------------- //
    // ------------------------------ TEMPLATE ------------------------------ //
    // ---------------------------------------------------------------------- //

    /**
     * Classical Binary Search - Template
     */
    public int binarySearch(int[] nums, int target) {
        if (nums == null || nums.length == 0) { // length == 0 is not necessary
            return -1;
        }

        int start = 0;
        int end = nums.length - 1;

        while (start <= end) { // use "<" if search for a range
            int mid = (start + end) >>> 1; // be aware of length of 2

            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                start = mid + 1;
            } else {
                end = mid - 1;
            }
        }

        return -1;
    }

    // ---------------------------------------------------------------------- //
    // ------------------------------ PROBLEMS ------------------------------ //
    // ---------------------------------------------------------------------- //

    /**
     * Search Insert Position.
     *
     * Given a sorted array and a target value, return the index if the target
     * is found. If not, return the index where it would be if it were inserted
     * in order. You may assume no duplicates in the array.
     *
     * Example: [1,3,5,6], 5 ¡ú 2. [1,3,5,6], 2 ¡ú 1. [1,3,5,6], 7 ¡ú 4. [1,3,5,6],
     * 0 ¡ú 0.
     *
     * Challenge: O(log(n)) time
     */
    @tags.Array
    @tags.SortedArray
    @tags.BinarySearch
    public int searchInsert(int[] A, int target) {
        if (A == null || A.length == 0) {
            return 0;
        }

        int start = 0, end = A.length - 1;

        while (start < end) {
            int mid = (start + end) >>> 1;

            if (A[mid] >= target) {
                end = mid;
            } else {
                start = mid + 1;
            }
        }

        return A[start] < target ? start + 1 : start;
    }

    /**
     * Search for a Range.
     *
     * Given a sorted array of integers, find the starting and ending position
     * of a given target value. If the target is not found in the array, return
     * [-1, -1].
     *
     * Example: Given [5, 7, 7, 8, 8, 10] and target value 8, return [3, 4].
     *
     * Challenge: O(log n) time.
     *
     * @param A
     *            : an integer sorted array
     * @param target
     *            : an integer to be inserted return : a list of length 2,
     *            [index1, index2]
     */
    @tags.Array
    @tags.SortedArray
    @tags.BinarySearch
    @tags.Company.LinkedIn
    @tags.Status.OK
    public int[] searchRange(int[] A, int target) {
        int[] range = new int[2];
        range[0] = -1;
        range[1] = -1;
        if (A == null || A.length == 0) {
            return range;
        }

        // search start
        int start = 0, end = A.length - 1;
        while (start < end) {
            int mid = (start + end) >>> 1; // left middle

            if (A[mid] < target) {
                start = mid + 1;
            } else if (A[mid] > target) {
                end = mid - 1;
            } else {
                end = mid;
            }
        }

        // add valid start or return
        if (A[start] == target) {
            range[0] = start;
        } else {
            return range;
        }

        // search end
        end = A.length - 1;
        while (start < end) {
            int mid = (start + end + 1) >>> 1; // right middle

            if (A[mid] == target) {
                start = mid;
            } else {
                end = mid - 1;
            }
        }

        // add end
        range[1] = end;

        return range;
    }

    /**
     * Search a 2D Matrix.
     *
     * Write an efficient algorithm that searches for a value in an m x n
     * matrix. This matrix has the following properties: Integers in each row
     * are sorted from left to right. The first integer of each row is greater
     * than the last integer of the previous row.
     *
     * Example: Consider the following matrix: [ [1, 3, 5, 7], [10, 11, 16, 20],
     * [23, 30, 34, 50] ] Given target = 3, return true.
     *
     * Challenge: O(log(n) + log(m)) time.
     */
    @tags.Matrix
    @tags.BinarySearch
    @tags.Status.Easy
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0] == null
                || matrix[0].length == 0) {
            return false;
        }

        int m = matrix.length, n = matrix[0].length;
        int start = 0, end = m * n - 1;

        while (start <= end) {
            int mid = (start + end) >>> 1;
            int row = mid / n, col = mid % n;

            if (matrix[row][col] > target) {
                end = mid - 1;
            } else if (matrix[row][col] < target) {
                start = mid + 1;
            } else {
                return true;
            }
        }

        return false;
    }

    /**
     * Search a 2D Matrix II.
     *
     * Write an efficient algorithm that searches for a value in an m x n
     * matrix, return the occurrence of it.
     *
     * This matrix has the following properties:
     * Integers in each row are sorted from left to right. Integers in each
     * column are sorted from up to bottom. No duplicate integers in each row or
     * column.
     *
     * This is not a binary search version, worst case complexity can be
     * discussed. Time complexity: O(m + n).
     *
     * @param matrix:
     *            A list of lists of integers
     * @param: A
     *             number you want to search in the matrix
     * @return: An integer indicate the occurrence of target in the given matrix
     */
    @tags.Matrix
    @tags.SortedMatrix
    @tags.BinarySearch
    @tags.Company.Amazon
    @tags.Company.Apple
    @tags.Company.Google
    @tags.Status.Easy
    public int searchMatrix2(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0] == null
                || matrix[0].length == 0) {
            return 0;
        }

        int m = matrix.length, n = matrix[0].length;
        int row = 0, col = n - 1;
        int count = 0;

        while (row < m && col >= 0) {
            if (matrix[row][col] < target) {
                row++;
            } else if (matrix[row][col] > target) {
                col--;
            } else {
                count++;
                row++;
                col--;
            }
        }

        return count;
    }

    /**
     * First Position of Target.
     *
     * For a given sorted array (ascending order) and a target number, find the
     * first index of this number in O(log n) time complexity.
     *
     * If the target number does not exist in the array, return -1.
     *
     * Example: If the array is [1, 2, 3, 3, 4, 5, 10], for given target 3,
     * return 2.
     *
     * Challenge: If the count of numbers is bigger than 2^32, can your code
     * work properly?
     *
     * @param nums:
     *            The integer array.
     * @param target:
     *            Target to find.
     * @return: The first position of target. Position starts from 0.
     */
    @tags.Array
    @tags.BinarySearch
    public int firstPosition(int[] nums, int target) {
        // write your code here
        if (nums == null || nums.length == 0) {
            return -1;
        }

        int start = 0, end = nums.length - 1;
        while (start < end) {
            int mid = (start + end) >>> 1;
            if (nums[mid] == target) {
                end = mid;
            } else if (nums[mid] > target) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }

        return (nums[start] == target) ? start : -1;
    }

    /**
     * Last Position of Target.
     *
     * Find the last position of a target number in a sorted array. Return -1 if
     * target does not exist.
     *
     * @param A
     *            an integer array sorted in ascending order
     * @param target
     *            an integer
     * @return an integer
     */
    public int lastPosition(int[] A, int target) {
        if (A == null || A.length == 0) {
            return -1;
        }

        int start = 0, end = A.length - 1;
        while (start < end) {
            int mid = (start + end + 1) >>> 1;
            if (A[mid] == target) {
                start = mid;
            } else if (A[mid] < target) {
                start = mid + 1;
            } else {
                end = mid - 1;
            }
        }

        if (start == end && A[start] == target) {
            return start;
        }
        return -1;
    }

    /**
     * Search in a Big Sorted Array.
     *
     * Given a big sorted array with positive integers sorted by ascending
     * order. The array is so big so that you can not get the length of the
     * whole array directly, and you can only access the kth number by
     * ArrayReader.get(k) (or ArrayReader->get(k) for C++). Find the first index
     * of a target number.
     *
     * Your algorithm should be in O(log k), where k is the first index of the
     * target number.
     *
     * Return -1, if the number doesn't exist in the array.
     *
     * Notice: If you accessed an inaccessible index (outside of the array),
     * ArrayReader.get will return 2,147,483,647.
     *
     * @param reader:
     *            An instance of ArrayReader. (Replaced by List here)
     * @param target:
     *            An integer
     * @return : An integer which is the index of the target number
     */
    public int searchBigSortedArray(List<Integer> reader, int target) {
        // write your code here
        int start;
        int end;

        // find the search end
        for (end = 1; end > 0; end = end << 1) {
            if (reader.get(end) >= target) {
                break;
            }
        }

        // get search range
        start = end >>> 1;
        if (end < 0) {
            end = Integer.MAX_VALUE;
        }

        while (start < end) {
            int mid = (start + end) >>> 1;
            int num = reader.get(mid);

            if (num == target) {
                end = mid;
            } else if (num < target) {
                start = mid + 1;
            } else {
                end = mid - 1;
            }
        }

        return (reader.get(start) == target) ? start : -1;
    }

    /**
     * Find Peak Element.
     *
     * There is an integer array which has the following features:
     *
     * The numbers in adjacent positions are different. A[0] < A[1] &&
     * A[A.length - 2] > A[A.length - 1]. We define a position P is a peek if:
     *
     * A[P] > A[P-1] && A[P] > A[P+1] Find a peak element in this array. Return
     * the index of the peak.
     *
     * Notice: The array may contains multiple peeks, find any of them.
     *
     * Example: Given [1, 2, 1, 3, 4, 5, 7, 6]. Return index 1 (which is number
     * 2) or 6 (which is number 7)
     *
     * Challenge: Time complexity O(logN)
     *
     * @param A:
     *            An integers array.
     * @return: return any of peek positions.
     */
    @tags.Array
    @tags.BinarySearch
    @tags.Source.LintCode
    @tags.Company.Google
    public int findPeak(int[] A) {
        int start = 0, end = A.length - 1;

        while (start < end) {
            int mid = (start + end) >>> 1;

            if (A[mid] > A[mid + 1]) {
                if (A[mid] > A[mid - 1]) {
                    return mid;
                } else {
                    end = mid - 1;
                }
            } else {
                start = mid + 1;
            }
        }

        return start;
    }

    /**
     * Find Peak Element II - O(nlogm) time.
     *
     * There is an integer matrix which has the following features: The numbers
     * in adjacent positions are different. The matrix has n rows and m columns.
     * For all i < m, A[0][i] < A[1][i] && A[n - 2][i] > A[n - 1][i]. For all j
     * < n, A[j][0] < A[j][1] && A[j][m - 2] > A[j][m - 1]. We define a position
     * P is a peek if: A[j][i] > A[j+1][i] && A[j][i] > A[j-1][i] && A[j][i] >
     * A[j][i+1] && A[j][i] > A[j][i-1] Find a peak element in this matrix.
     * Return the index of the peak.
     *
     * Notice: The matrix may contains multiple peeks, find any of them.
     *
     * Example: Given a matrix: [ [1 ,2 ,3 ,6 ,5], [16,41,23,22,6],
     * [15,17,24,21,7], [14,18,19,20,10], [13,14,11,10,9] ], return index of 41
     * (which is [1,1]) or index of 24 (which is [2,2]).
     *
     * Challenge: Solve it in O(n+m) time. If you come up with an algorithm that
     * you thought it is O(n log m) or O(m log n), can you prove it is actually
     * O(n+m) or propose a similar but O(n+m) algorithm?
     *
     * Check the link for O(m + n) solution:
     * http://courses.csail.mit.edu/6.006/spring11/lectures/lec02.pdf.
     *
     * TODO understand the solution.
     *
     * @param A:
     *            An integer matrix
     * @return: The index of the peak
     */
    @tags.BinarySearch
    @tags.Matrix
    @tags.Source.LintCode
    public List<Integer> findPeakII(int[][] A) {
        List<Integer> index = new ArrayList<>();
        if (A == null || A.length == 0 || A[0] == null || A[0].length == 0) {
            return index;
        }

        int start = 1, end = A.length - 2;
        while (start <= end) {
            int mid = (start + end) >>> 1;
            int col = findMax(A[mid]);
            if (A[mid][col] < A[mid - 1][col]) {
                end = mid - 1;
            } else if (A[mid][col] < A[mid + 1][col]) {
                start = mid + 1;
            } else {
                index.add(mid);
                index.add(col);
                return index;
            }
        }

        return index;
    }

    private int findMax(int[] nums) {
        int index = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > nums[index]) {
                index = i;
            }
        }
        return index;
    }

    /**
     * First Bad Version.
     *
     * The code base version is an integer start from 1 to n. One day, someone
     * committed a bad version in the code case, so it caused this version and
     * the following versions are all failed in the unit tests. Find the first
     * bad version.
     *
     * You can call isBadVersion to help you determine which version is the
     * first bad one. The details interface can be found in the code's
     * annotation part.
     *
     * @param n
     *            An integers.
     * @return An integer which is the first bad version.
     */
    @tags.BinarySearch
    @tags.Source.LeetCode
    @tags.Source.LintCode
    @tags.Company.Facebook
    @tags.Status.Easy
    public int findFirstBadVersion(int n) {
        if (n < 1) {
            return -1;
        }

        int start = 1, end = n;

        while (start <= end) {
            int mid = (start + end) >>> 1;

            if (SVNRepo.isBadVersion(mid)) {
                end = mid;
            } else {
                start = mid + 1;
            }
        }

        return SVNRepo.isBadVersion(start) ? start : -1;
    }

    /**
     * Helper class just for testing. Call {@link #setBadVersion(int)} before
     * using it.
     */
    public static class SVNRepo {
        static int badVersion = 1;

        public static void setBadVersion(int badVersion) {
            SVNRepo.badVersion = badVersion;
        }

        private static boolean isBadVersion(int n) {
            return badVersion == n;
        }
    }

    /**
     * Total Occurrence of Target.
     *
     * Given a target number and an integer array sorted in ascending order.
     * Find the total number of occurrences of target in the array.
     *
     * @param A
     *            an integer array sorted in ascending order
     * @param target
     *            an integer
     * @return an integer
     */
    public int totalOccurrence(int[] A, int target) {
        if (A == null || A.length == 0) {
            return 0;
        }

        int firstIndex;

        // find first index

        int start = 0, end = A.length - 1;
        while (start < end) {
            int mid = (start + end) >>> 1; // left middle
            if (A[mid] == target) {
                end = mid;
            } else if (A[mid] > target) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }

        if (A[start] == target) {
            firstIndex = start;
        } else {
            return 0;
        }

        // find last index

        end = A.length - 1;
        while (start < end) {
            int mid = (start + end + 1) >>> 1; // right middle
            if (A[mid] == target) {
                start = mid;
            } else {
                end = mid - 1;
            }
        }

        return end - firstIndex + 1;
    }

    /**
     * Closest Number in Sorted Array.
     *
     * Given a target number and an integer array A sorted in ascending order,
     * find the index i in A such that A[i] is closest to the given target.
     *
     * Return -1 if there is no element in the array.
     *
     * Notice: There can be duplicate elements in the array, and we can return
     * any of the indices with same value.
     *
     * @param A
     *            an integer array sorted in ascending order
     * @param target
     *            an integer
     * @return an integer
     */
    public int closestNumber(int[] A, int target) {
        if (A == null || A.length < 1) {
            return -1;
        }

        int start = 0, end = A.length - 1;

        while (start + 1 < end) {
            int mid = (start + end) >>> 1;

            if (A[mid] == target) {
                return mid;
            } else if (A[mid] < target) {
                start = mid;
            } else {
                end = mid;
            }
        }

        if (Math.abs(A[start] - target) <= Math.abs(A[end] - target)) {
            return start;
        }

        return end;
    }

    /**
     * K Closest Numbers In Sorted Array.
     *
     * Given a target number, a non-negative integer k and an integer array A
     * sorted in ascending order, find the k closest numbers to target in A,
     * sorted in ascending order by the difference between the number and
     * target. Otherwise, sorted in ascending order by number if the difference
     * is same.
     *
     * @param A
     *            an integer array
     * @param target
     *            an integer
     * @param k
     *            a non-negative integer
     * @return an integer array
     */
    public int[] kClosestNumbers(int[] A, int target, int k) {
        int[] result = new int[k];
        if (A == null || k <= 0 || A.length < k) {
            return result;
        }

        int start = 0, end = A.length - 1;
        int min = -1;

        while (start + 1 < end) {
            int mid = (start + end) >>> 1;

            if (A[mid] == target) {
                min = mid;
                break;
            } else if (A[mid] < target) {
                start = mid;
            } else {
                end = mid;
            }
        }

        if (min == -1) {
            if (Math.abs(A[start] - target) <= Math.abs(A[end] - target)) {
                min = start;
            } else {
                min = end;
            }
        }

        // compare left and right
        result[0] = A[min];
        int index = 1;
        start = min;
        end = min;
        while (index < k) {
            int next = -1;
            if (start - 1 >= 0) {
                start = start - 1;
                next = start;
            }

            if (end + 1 < A.length) {
                if (next == -1) {
                    end = end + 1;
                    next = end;
                } else {
                    int startDiff = Math.abs(A[start] - target);
                    int endDiff = Math.abs(A[end + 1] - target);
                    if (startDiff > endDiff) {
                        start++;
                        end++;
                        next = end;
                    }
                }
            }

            if (next == -1) {
                return result;
            }

            result[index] = A[next];
            index++;
        }

        return result;
    }

    /**
     * Wood Cut.
     *
     * Given n pieces of wood with length L[i] (integer array). Cut them into
     * small pieces to guarantee you could have equal or more than k pieces with
     * the same length. What is the longest length you can get from the n pieces
     * of wood? Given L & k, return the maximum length of the small pieces.
     *
     * Notice: You couldn't cut wood into float length.
     *
     * Example: For L=[232, 124, 456], k=7, return 114.
     *
     * Challenge: O(n log Len), where Len is the longest length of the wood.
     *
     * @param L:
     *            Given n pieces of wood with length L[i]
     * @param k:
     *            An integer
     * @return: The maximum length of the small pieces.
     */
    @tags.BinarySearch
    public int woodCut(int[] L, int k) {
        if (L == null || L.length == 0 || k <= 0) {
            return 0;
        }

        // find the longest
        int max = L[0];
        for (int i = 1; i < L.length; i++) {
            if (L[i] > max) {
                max = L[i];
            }
        }

        int start = 1, end = max;
        while (start < end) {
            int mid = (start + end + 1) >>> 1;

            if (countPieces(L, mid) >= k) {
                start = mid;
            } else {
                end = mid - 1;
            }
        }

        // not enough wood to cut k pieces out
        if (countPieces(L, start) >= k) {
            return start;
        }
        return 0;
    }

    private int countPieces(int[] L, int len) {
        int count = 0;
        for (int i = 0; i < L.length; i++) {
            count += L[i] / len;
        }
        return count;
    }

    // ---------------------------------------------------------------------- //
    // ------------------------------ ROTATED ------------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Find Minimum in Rotated Sorted Array.
     *
     * Suppose a sorted array is rotated at some pivot unknown to you
     * beforehand. Find the minimum element.
     *
     * You may assume no duplicate exists in the array.
     *
     * Example: Given [4, 5, 6, 7, 0, 1, 2] return 0
     *
     * @param nums: a rotated sorted array
     * @return: the minimum number in the array
     */
    @tags.BinarySearch
    @tags.Status.OK
    public int findMin(int[] num) {
        if (num == null || num.length == 0) {
            throw new IllegalArgumentException();
        }

        int start = 0, end = num.length - 1;

        while (start < end) {
            int mid = (start + end) >>> 1;

            if (num[mid] < num[end]) {
                end = mid;
            } else {
                start = mid + 1;
            }
        }

        return num[start];
    }

    /**
     * Find Minimum in Rotated Sorted Array II.
     *
     * Suppose a sorted array is rotated at some pivot unknown to you
     * beforehand. (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2). Find the
     * minimum element.
     *
     * Notice: The array may contain duplicates.
     *
     * Example: Given [4,4,5,6,7,0,1,2] return 0.
     *
     * @param num:
     *            a rotated sorted array
     * @return: the minimum number in the array
     */
    @tags.BinarySearch
    @tags.DivideAndConquer
    @tags.Status.OK
    public int findMinWithDup(int[] num) {
        if (num == null || num.length == 0) {
            throw new IllegalArgumentException();
        }

        int start = 0, end = num.length - 1;
        while (start < end) {
            int mid = (start + end) >>> 1;
            if (num[mid] < num[end]) {
                end = mid;
            } else if (num[mid] > num[end]) {
                start = mid + 1;
            } else {
                end--;
            }
        }

        return num[start];
    }

    /**
     * Search in Rotated Sorted Array
     *
     * Suppose a sorted array is rotated at some pivot unknown to you
     * beforehand. (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2). You are
     * given a target value to search. If found in the array return its index,
     * otherwise return -1. You may assume no duplicate exists in the array.
     *
     * Example: For [4, 5, 1, 2, 3] and target=1, return 2. For [4, 5, 1, 2, 3]
     * and target=0, return -1.
     *
     * Challenge: O(logN) time.
     *
     * @param A
     *            : an integer rotated sorted array
     * @param target
     *            : an integer to be searched return : an integer
     */
    @tags.BinarySearch
    @tags.Array
    @tags.SortedArray
    @tags.Company.Facebook
    @tags.Company.LinkedIn
    @tags.Company.Uber
    @tags.Status.NeedPractice
    public int search(int[] A, int target) {
        if (A == null || A.length == 0) {
            return -1;
        }

        int start = 0, end = A.length - 1;

        while (start <= end) {
            int mid = (start + end) >>> 1;
            if (A[mid] < target) {
                if (A[start] > A[mid] && A[end] < target) {
                    end = mid - 1;
                } else {
                    start = mid + 1;
                }
            } else if (A[mid] > target) {
                if (A[mid] > A[end] && A[start] > target) {
                    start = mid + 1;
                } else {
                    end = mid - 1;
                }
            } else {
                return mid;
            }
        }

        return -1;
    }

    /**
     * Search in Rotated Sorted Array II
     *
     * Follow up for "Search in Rotated Sorted Array": What if duplicates are
     * allowed? Would this affect the run-time complexity? How and why?
     *
     * Write a function to determine if a given target is in the array.
     *
     * Example: Given [1, 1, 0, 1, 1, 1] and target = 0, return true. Given [1,
     * 1, 1, 1, 1, 1] and target = 0, return false.
     *
     * O(logN) ~ O(n), depends on number of duplicates. This solutions is so
     * concise and beautiful.
     *
     * @param A
     *            : an integer ratated sorted array and duplicates are allowed
     * @param target
     *            : an integer to be search
     * @return : a boolean
     */
    @tags.BinarySearch
    @tags.Array
    @tags.SortedArray
    public boolean searchII(int[] A, int target) {
        if (A == null || A.length == 0) {
            return false;
        }

        int start = 0, end = A.length - 1;

        while (start <= end) {
            int mid = (start + end) >>> 1;

            if (A[mid] == target) {
                return true; // or return index according to requirement;
            }

            if (A[mid] < A[end]) { // right part is sorted
                if (A[mid] < target && A[end] >= target) {
                    start = mid + 1;
                } else {
                    end = mid - 1;
                }
            } else if (A[mid] > A[end]) { // left part is sorted
                if (A[mid] < target && A[start] >= target) {
                    end = mid - 1;
                } else {
                    start = mid + 1;
                }
            } else {
                end--;
            }
        }

        return false;
    }

    // ------------------------------- OLD ---------------------------------

    /**
     * 
     */
    // Solution to copy books, the following code is in c++ since wikioi
    // only accetps c++
    // #include <stdio.h>
    // #include <string.h>
    // #include <stdlib.h>
    // #include <algorithm>
    // #include <iostream>
    // #include <cstdio>
    // using namespace std;
    //
    // // Check whether a given number of pages in a slice is
    // // valid, i.e. all the books could get copied.
    // bool isValid(int M, int K, int* pages, int sliceNum) {
    // int curSliceNum = 0;
    // int curBook = M - 1;
    // for(int i = K - 1; i >= 0; i--) {
    // curSliceNum = 0;
    //
    // while(curSliceNum + pages[curBook] <= sliceNum &&
    // curBook >= 0) {
    // curSliceNum += pages[curBook];
    // curBook--;
    // }
    //
    // if (curBook < 0) {
    // return true;
    // }
    // }
    //
    // return false;
    // }
    //
    //
    // // Use binary search to find the optimal number of pages in a slice.
    // int search(int M, int K, int* pages, int minSliceNum, int maxSliceNum) {
    // int beg = minSliceNum;
    // int end = maxSliceNum;
    // int mid;
    //
    // while (beg + 1 < end) {
    // mid = (beg + end) / 2;
    // if (isValid(M, K, pages, mid)) {
    // end = mid;
    // } else {
    // beg = mid;
    // }
    // }
    //
    // if (isValid(M, K, pages, end)) {
    // return end;
    // }
    //
    // return beg;
    // }
    //
    // int main() {
    // int M, K;
    // scanf("%d %d", &M, &K);
    //
    // int* pages = new int[M];
    // int* startBook = new int[K];
    // int* endBook = new int[K];
    // int maxSliceNum = 0;
    // int minSliceNum = 0;
    // int optimalSliceNum;
    // for(int i = 0; i < M; i++) {
    // scanf("%d ", &pages[i]);
    // minSliceNum = min(pages[i], minSliceNum);
    // maxSliceNum += pages[i];
    // }
    //
    // optimalSliceNum = search(M, K, pages, minSliceNum, maxSliceNum);
    //
    // int curSliceNum = 0;
    // int curBook = M - 1;
    // for(int i = K - 1; i >= 0; i--) {
    // curSliceNum = 0;
    // endBook[i] = curBook;
    // while (curSliceNum + pages[curBook] <= optimalSliceNum &&
    // curBook >= i) {
    // curSliceNum += pages[curBook];
    // curBook--;
    // }
    // startBook[i] = curBook + 1;
    // }
    //
    // for(int i = 0; i < K; i++) {
    // printf("%d %d\n", startBook[i] + 1, endBook[i] + 1);
    // }
    //
    // delete [] endBook;
    // delete [] startBook;
    // delete [] pages;
    //
    // return 0;
    // }

    @Test
    public void test() {
        int[] nums = new int[3];
        nums[0] = 1;
        nums[1] = 2;
        nums[2] = 3;
        System.out.println(binarySearch(nums, 1));

        System.out.println(kClosestNumbers(nums, 2, 3));

        woodCutTest();
    }

    private void woodCutTest() {
        int[] L = {232,124,456};
        int k = 7;
        Assert.assertEquals(124, woodCut(L, k));
    }

}
