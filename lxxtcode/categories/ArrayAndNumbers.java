package categories;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;

public class ArrayAndNumbers {

    // ------------------------------- PROBLEMS --------------------------------

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
     * Two Sum Closest
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

    // ---------------------------------- OLD ----------------------------------

    /**
     * 3Sum
     *
     * Given an array S of n integers, are there elements a, b, c in S such that
     * a + b + c = 0? Find all unique triplets in the array which gives the sum
     * of zero.
     *
     * Note: Elements in a triplet (a,b,c) must be in non-descending order. (ie,
     * a ¡Ü b ¡Ü c) The solution set must not contain duplicate triplets. For
     * example, given array S = {-1 0 1 2 -1 -4},
     *
     * A solution set is: (-1, 0, 1) (-1, -1, 2)
     */
    public ArrayList<ArrayList<Integer>> threeSum(int[] num) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
        Arrays.sort(num);

        int len = num.length;
        for (int i = 0; i < len; i++) {
            int first = num[i];
            if (i > 0 && first == num[i - 1])
                continue;

            int start = i + 1;
            int end = len - 1;

            while (start < end) {
                int sum = first + num[start] + num[end];
                if (sum == 0) {
                    // add result
                    ArrayList<Integer> three = new ArrayList<Integer>();
                    three.add(first);
                    three.add(num[start]);
                    three.add(num[end]);
                    result.add(three);

                    // shrink range and skip duplicate
                    start++;
                    end--;
                    while (start < end && num[start] == num[start - 1])
                        start++;
                    while (start < end && num[end] == num[end + 1])
                        end--;
                } else if (sum > 0) {
                    end--;
                    while (start < end && num[end] == num[end + 1])
                        end--;
                } else {
                    start++;
                    while (start < end && num[start] == num[start - 1])
                        start++;
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
     * For example, given array S = {-1 2 1 -4}, and target = 1.
     *
     * The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
     */
    public int threeSumClosest(int[] num, int target) {
        Arrays.sort(num);

        int sum = num[0] + num[1] + num[2];

        // traverse the array for every possible position of first number
        for (int i = 0; i < num.length - 2; i++) {
            // second number start from the one next to first one
            // third number start from the last number in the array
            for (int j = i + 1, k = num.length - 1; j < k;) {
                int temp = num[i] + num[j] + num[k];

                // compare temp with target
                if (temp == target) {
                    return temp;
                } else {
                    // update sum
                    if (Math.abs(temp - target) < Math.abs(sum - target)) {
                        sum = temp;
                    }

                    // update j and k
                    if (temp > target) {
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
     * 4Sum
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
     */
    public ArrayList<ArrayList<Integer>> fourSum(int[] num, int target) {
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

    public void test() {
        int[] A = { 2, 3, 4, 2, 5 };
        System.out.println(findDup(A));

        System.out.println(10 / 2);
        System.out.println(10 >>> 1);
    }

    public static void main(String[] args) {
        ArrayAndNumbers an = new ArrayAndNumbers();
        an.test();
    }

}
