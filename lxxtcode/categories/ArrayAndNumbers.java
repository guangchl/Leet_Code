package categories;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;

import org.junit.Test;
import org.junit.Assert;

/**
 * Arrays and Numbers.
 *
 * @author Guangcheng Lu
 */
public class ArrayAndNumbers {

    // ---------------------------------------------------------------------- //
    // ------------------------------- PROBLEMS ----------------------------- //
    // ---------------------------------------------------------------------- //

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
     * Merge Sorted Array
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
    @tags.SortedArray
    @tags.Company.Facebook
    public void mergeSortedArray(int[] A, int m, int[] B, int n) {
        if (A == null || B == null || A.length < m + n) {
            return;
        }

        while (m > 0 || n > 0) {
            if (n == 0 || (m > 0 && A[m - 1] >= B[n - 1])) {
                A[m + n - 1] = A[m - 1];
                m--;
            } else {
                A[m + n - 1] = B[n - 1];
                n--;
            }
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
     * Partition Array
     *
     * Given an array nums of integers and an int k, partition the array (i.e
     * move the elements in "nums") such that: All elements < k are moved to the
     * left All elements >= k are moved to the right Return the partitioning
     * index, i.e the first index i nums[i] >= k.
     *
     * You should do really partition in array nums instead of just counting the
     * numbers of integers smaller than k.
     *
     * If all elements in nums are smaller than k, then return nums.length
     *
     * Example: If nums = [3,2,2,1] and k=2, a valid answer is 1.
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

        int left = 0, right = nums.length - 1;
        while (left < right) {
            if (nums[left] < k) {
                left++;
            } else {
                int temp = nums[right];
                nums[right] = nums[left];
                nums[left] = temp;
                right--;
            }
        }

        if (nums[left] < k) {
            return left + 1;
        } else {
            return left;
        }
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
    public int kthLargestElement(int k, int[] nums) {
        if (nums == null || nums.length < k) {
            throw new IllegalArgumentException();
        }
        return getKthNumber(nums, 0, nums.length - 1, k);
    }

    private int getKthNumber(int[] nums, int start, int end, int k) {
        int pivot = partition(nums, start, end);
        if (pivot + 1 == k) {
            return nums[pivot];
        } else if (pivot + 1 < k) {
            return getKthNumber(nums, pivot + 1, end, k);
        } else {
            return getKthNumber(nums, start, pivot - 1, k);
        }
    }

    private int partition(int[] nums, int start, int end) {
        int pivot = start;
        while (start < end) {
            while (start < end && nums[end] <= nums[pivot]) {
                end--;
            }
            while (start < end && nums[start] >= nums[pivot]) {
                start++;
            }

            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
        }
        int temp = nums[start];
        nums[start] = nums[pivot];
        nums[pivot] = temp;
        return start;
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
    @tags.Company.Google
    @tags.Company.Uber
    @tags.Company.Zenefits
    public double findMedianSortedArrays(int[] A, int[] B) {
        if (A == null || B == null) {
            return 0;
        }
        int len = A.length + B.length;
        if (len % 2 == 1) {
            return findKth(A, 0, B, 0, len / 2 + 1);
        } else {
            int left = findKth(A, 0, B, 0, len / 2);
            int right = findKth(A, 0, B, 0, len / 2 + 1);
            return (left + right) / 2.0;
        }
    }

    private int findKth(int[] A, int aStart, int[] B, int bStart, int k) {
        if (aStart >= A.length) {
            return B[bStart + k - 1];
        }
        if (bStart >= B.length) {
            return A[aStart + k - 1];
        }

        if (k == 1) {
            return Math.min(A[aStart], B[bStart]);
        }

        int aNum = aStart + k / 2 - 1 < A.length ? A[aStart + k / 2 - 1]
                : Integer.MAX_VALUE;
        int bNum = bStart + k / 2 - 1 < B.length ? B[bStart + k / 2 - 1]
                : Integer.MAX_VALUE;

        if (aNum < bNum) {
            return findKth(A, aStart + k / 2, B, bStart, k - k / 2);
        } else {
            return findKth(A, aStart, B, bStart + k / 2, k - k / 2);
        }
    }

    // ---------------------------------------------------------------------- //
    // ------------------------------ Subarray ------------------------------ //
    // ---------------------------------------------------------------------- //

    /**
     * Subarray Sum
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
     * Subarray Sum Closest
     *
     * Given an integer array, find a subarray with sum closest to zero. Return
     * the indexes of the first number and last number.
     *
     * Example: Given [-3, 1, 1, -3, 5], return [0, 2], [1, 3], [1, 1], [2, 2]
     * or [0, 4].
     *
     * Time complexity: O(nlogn).
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

        int sum = 0, index = -1;
        int[] sums = new int[nums.length + 1]; // sums[i] = prefix sum at i
        sums[nums.length] = sum;
        Map<Integer, Integer> map = new HashMap<>(); // prefix sum to index map
        map.put(sum, index);

        // find prefix sum for each index
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            sums[i] = sum;

            if (map.containsKey(sum)) {
                range[0] = map.get(sum) + 1;
                range[1] = i;
                return range;
            }
            map.put(sum, i);
        }

        Arrays.sort(sums);

        // find the 2 closest prefix sum, put the 2 index in result
        int diff = Math.abs(sums[0] - sums[1]);
        range[0] = map.get(sums[0]);
        range[1] = map.get(sums[1]);
        for (int i = 1; i < sums.length; i++) {
            int newDiff = sums[i] - sums[i - 1];
            if (newDiff < diff) {
                diff = newDiff;
                range[0] = map.get(sums[i]);
                range[1] = map.get(sums[i - 1]);
            }
        }

        Arrays.sort(range);
        range[0] += 1;
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
     * Maximum Subarray
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
    @tags.Company.LinkedIn
    @tags.Source.LintCode
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
    public int maxSubArray(int[] nums, int k) {
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
    @tags.Source.LintCode
    public int maxDiffSubArrays(int[] nums) {
        int len = nums.length;
        int[] forwardMax = new int[len];
        int[] forwardMin = new int[len];
        int maxToHere = nums[0];
        int minToHere = nums[0];
        forwardMax[0] = nums[0];
        forwardMin[0] = nums[0];
        for (int i = 1; i < len; i++) {
            maxToHere = Math.max(nums[i], nums[i] + maxToHere);
            forwardMax[i] = Math.max(forwardMax[i - 1], maxToHere);
            minToHere = Math.min(nums[i], nums[i] + minToHere);
            forwardMin[i] = Math.min(forwardMin[i - 1], minToHere);
        }

        int[] backwardMax = new int[len];
        int[] backwardMin = new int[len];
        maxToHere = nums[len - 1];
        minToHere = nums[len - 1];
        backwardMax[len - 1] = nums[len - 1];
        backwardMin[len - 1] = nums[len - 1];
        for (int i = len - 2; i >= 0; i--) {
            maxToHere = Math.max(nums[i], nums[i] + maxToHere);
            backwardMax[i] = Math.max(backwardMax[i + 1], maxToHere);
            minToHere = Math.min(nums[i], nums[i] + minToHere);
            backwardMin[i] = Math.min(backwardMin[i + 1], minToHere);
        }

        int diff = 0;
        for (int i = 0; i < len - 1; i++) {
            diff = Math.max(diff, Math.abs(forwardMax[i] - backwardMin[i + 1]));
            diff = Math.max(diff, Math.abs(forwardMin[i] - backwardMax[i + 1]));
        }

        return diff;
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
    public int minSubArray(ArrayList<Integer> nums) {
        if (nums == null || nums.size() == 0) {
            return -1;
        }
        int min = nums.get(0);
        int minToHere = nums.get(0);

        for (int i = 1; i < nums.size(); i++) {
            minToHere = (minToHere < 0) ? minToHere + nums.get(i) : nums.get(i);
            if (minToHere < min) {
                min = minToHere;
            }
        }

        return min;
    }

    /**
     * Maximum Product Subarray
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

    // ---------------------------------------------------------------------- //
    // ------------------------------ Two Sum ------------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Two Sum
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
    @tags.Company.Airbnb
    @tags.Company.Facebook
    public int[] twoSum(int[] numbers, int target) {
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
     * 3Sum
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
    @tags.Company.Facebook
    public ArrayList<ArrayList<Integer>> threeSum(int[] numbers) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();

        Arrays.sort(numbers);

        for (int i = 0; i < numbers.length - 2; i++) {
            if (i != 0 && numbers[i] == numbers[i - 1]) {
                continue;
            }

            int j = i + 1, k = numbers.length - 1;
            int target = -numbers[i];
            while (j < k) {
                int twoSum = numbers[j] + numbers[k];
                if (twoSum == target) {
                    ArrayList<Integer> list = new ArrayList<>();
                    list.add(numbers[i]);
                    list.add(numbers[j]);
                    list.add(numbers[k]);
                    result.add(list);
                    j++;
                    k--;
                    while (j < k && numbers[j] == numbers[j - 1]) {
                        j++;
                    }
                    while (numbers[k] == numbers[k + 1]) {
                        k--;
                    }
                } else if (twoSum < target) {
                    j++;
                } else {
                    k--;
                }
            }
        }

        return result;
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
        for (int i = 0; i < numbers.length - 3; i++) {
            if (i != 0 && numbers[i] == numbers[i - 1]) {
                continue;
            }

            for (int j = i + 1; j < numbers.length - 2; j++) {
                if (j != i + 1 && numbers[j] == numbers[j - 1]) {
                    continue;
                }

                int twoSum = numbers[i] + numbers[j];
                for (int k = j + 1, l = numbers.length - 1; k < l;) {
                    int fourSum = twoSum + numbers[k] + numbers[l];
                    if (fourSum == target) {
                        ArrayList<Integer> list = new ArrayList<>();
                        list.add(numbers[i]);
                        list.add(numbers[j]);
                        list.add(numbers[k]);
                        list.add(numbers[l]);
                        result.add(list);
                        k++;
                        l--;
                        while (k < l && numbers[k] == numbers[k - 1]) {
                            k++;
                        }
                        while (numbers[l] == numbers[l + 1]) {
                            l--;
                        }
                    } else if (fourSum > target) {
                        l--;
                    } else {
                        k++;
                    }
                }
            }
        }

        return result;
    }

    /** 4Sum - O(n<sup>2</sup>logn<sup>2</sup>) = O(n<sup>2</sup>logn) time. */
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
     * Remove Duplicates from Sorted Array
     * 
     * @param A
     * @return new length
     */
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

    /**
     * Largest Rectangle in Histogram
     * 
     * Given n non-negative integers representing the histogram's bar height
     * where the width of each bar is 1, find the area of largest rectangle in
     * the histogram.
     */
    public int largestRectangleArea(int[] height) {
        int largest = 0;

        Stack<Integer> stack = new Stack<Integer>();
        int index = 0;
        while (index < height.length || !stack.isEmpty()) {
            if (index < height.length && (stack.isEmpty()
                    || height[index] >= height[stack.peek()])) {
                stack.push(index++);
            } else {
                int end = index - 1;
                int h = height[stack.pop()];
                while (!stack.isEmpty() && height[stack.peek()] == h) {
                    stack.pop();
                }
                int start = stack.isEmpty() ? -1 : stack.peek();
                largest = Math.max(largest, h * (end - start));
            }
        }

        return largest;
    }

    /**
     * Given a 2D binary matrix filled with 0's and 1's, find the largest
     * rectangle containing all ones and return its area.
     */
    public int maximalRectangle(char[][] matrix) {
        int m = matrix.length;
        if (m == 0)
            return 0;
        int n = matrix[0].length;
        if (n == 0)
            return 0;
        int[][] rectangles = new int[m][n];

        // pre-process first row
        for (int i = 0; i < n; i++) {
            if (matrix[0][i] == '1') {
                rectangles[0][i] = 1;
            }
        }

        // pre-process other rows
        for (int i = 1; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '1')
                    rectangles[i][j] = rectangles[i - 1][j] + 1;
            }
        }

        int area = 0;

        // calculate largest rectangle for each row
        for (int i = 0; i < m; i++) {
            area = Math.max(area, largestRectangleArea(rectangles[i]));
        }

        return area;
    }

    // ---------------------------------------------------------------------- //
    // ------------------------------ Unit Tests ---------------------------- //
    // ---------------------------------------------------------------------- //

    @Test
    public void test() {
        int[] findDup = { 2, 3, 4, 2, 5 };
        System.out.println(findDup(findDup));

        continuousSubarraySumTest();
        continuousSubarraySumIITest();
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

        int[] nums2 = {29,84,-44,17,-22,40,-5,19,90};
        range = continuousSubarraySumII(nums2);
        Assert.assertTrue(range.get(0) == 5 && range.get(1) == 1);

        int[] nums3 = {-5,10,5,-3,1,1,1,-2,3,-4};
        range = continuousSubarraySumII(nums3);
        Assert.assertTrue(range.get(0) == 1 && range.get(1) == 8);
    }

}
