package categories;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;

import org.junit.Assert;
import org.junit.Test;

public class MathAndBitManipulation {

    // ---------------------------------------------------------------------- //
    // ------------------------------- MODELS ------------------------------- //
    // ---------------------------------------------------------------------- //

    /** Definition for Point. */
    class Point {
        int x;
        int y;

        Point() {
            x = 0;
            y = 0;
        }

        Point(int a, int b) {
            x = a;
            y = b;
        }
    }

    // ---------------------------------------------------------------------- //
    // --------------------------------- MATH ------------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Evaluate Reverse Polish Notation.
     *
     * Evaluate the value of an arithmetic expression in Reverse Polish
     * Notation. Valid operators are +, -, *, /. Each operand may be an integer
     * or another expression.
     *
     * Example: ["2", "1", "+", "3", "*"] -> ((2 + 1) * 3) -> 9. ["4", "13",
     * "5", "/", "+"] -> (4 + (13 / 5)) -> 6.
     *
     * @param tokens
     *            The Reverse Polish Notation
     * @return the value
     */
    @tags.Stack
    @tags.Company.LinkedIn
    @tags.Company.Thumbtack
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();

        for (String token : tokens) {
            switch (token) {
            case "+":
                stack.push(stack.pop() + stack.pop());
                break;
            case "-":
                stack.push(-stack.pop() + stack.pop());
                break;
            case "*":
                stack.push(stack.pop() * stack.pop());
                break;
            case "/":
                int n1 = stack.pop(), n2 = stack.pop();
                stack.push(n2 / n1);
                break;
            default:
                stack.push(Integer.parseInt(token));
                break;
            }
        }

        return stack.pop();
    }

    /**
     * Trailing Zeros.
     *
     * Write an algorithm which computes the number of trailing zeros in n
     * factorial.
     *
     * Example: 11! = 39916800, so the out should be 2.
     *
     * Challenge: O(log N) time.
     *
     * @param n:
     *            As desciption
     * @return: An integer, denote the number of trailing zeros in n!
     */
    @tags.Math
    public long trailingZeros(long n) {
        long count = 0;
        while (n >= 5) {
            count += n / 5;
            n /= 5;
        }
        return count;
    }

    /**
     * Print Numbers by Recursion.
     *
     * Print numbers from 1 to the largest number with N digits by recursion.
     *
     * Notice: Can you do it in another way to recursive with at most N depth?
     *
     * Example: Given N = 1, return [1,2,3,4,5,6,7,8,9]. Given N = 2, return
     * [1,2,3,4,5,6,7,8,9,10,11,12,...,99].
     *
     * Challenge: Do it in recursion, not for-loop.
     *
     * @param n:
     *            An integer. return : An array storing 1 to the largest number
     *            with n digits.
     */
    @tags.Recursion
    @tags.Status.OK
    public List<Integer> numbersByRecursion(int n) {
        if (n == 0) {
            return new ArrayList<>();
        }

        List<Integer> result = numbersByRecursion(n - 1);
        int max = (int) Math.pow(10, n);
        for (int i = max / 10; i < max; i++) {
            result.add(i);
        }
        return result;
    }

    /**
     * Delete Digits.
     *
     * Given string A representative a positive integer which has N digits,
     * remove any k digits of the number, the remaining digits are arranged
     * according to the original order to become a new positive integer. Find
     * the smallest integer after remove k digits. N <= 240 and k <= N.
     *
     * Example: Given an integer A = "178542", k = 4. return a string "12".
     *
     * @param A:
     *            A positive integer which has N digits, A is a string.
     * @param k:
     *            Remove k digits.
     * @return: A string
     */
    @tags.Greedy
    @tags.Source.LintCode
    public String DeleteDigits(String A, int k) {
        StringBuilder sb = new StringBuilder(A);
        while (k > 0) {
            for (int i = 0; i < sb.length(); i++) {
                if (i == sb.length() - 1 || sb.charAt(i) > sb.charAt(i + 1)) {
                    sb.deleteCharAt(i);
                    k--;
                    break;
                }
            }
        }

        int i = 0;
        while (sb.charAt(i) == '0' && i < sb.length()) {
            i++;
        }
        return sb.substring(i);
    }

    /**
     * Reorder array to construct the minimum number.
     *
     * Construct minimum number by reordering a given non-negative integer
     * array. Arrange them such that they form the minimum number.
     *
     * Notice: The result may be very large, so you need to return a string
     * instead of an integer.
     *
     * Example: Given [3, 32, 321], there are 6 possible numbers can be
     * constructed by reordering the array: 3+32+321=332321, 3+321+32=332132,
     * 32+3+321=323321, 32+321+3=323213, 321+3+32=321332, 321+32+3=321323. So
     * after reordering, the minimum number is 321323, and return it.
     *
     * Challenge: Do it in O(nlogn) time complexity.
     *
     * @param nums
     *            n non-negative integer array
     * @return a string
     */
    @tags.Array
    @tags.Permutation
    @tags.Status.NeedPractice
    public String minNumber(int[] nums) {
        if (nums == null || nums.length == 0) {
            return "";
        }

        // convert numbers to strings
        List<String> strings = new ArrayList<>();
        for (Integer i : nums) {
            strings.add(String.valueOf(i));
        }

        // sort
        Collections.sort(strings, new Comparator<String>() {
            @Override
            public int compare(String s1, String s2) {
                StringBuilder sb1 = new StringBuilder(s1);
                while (sb1.length() < s2.length()) {
                    sb1.append(s1.charAt(0));
                }
                s1 = sb1.toString();

                StringBuilder sb2 = new StringBuilder(s2);
                while (s1.length() > sb2.length()) {
                    sb2.append(s2.charAt(0));
                }
                s2 = sb2.toString();
                return s1.compareTo(s2);
            }
        });

        // combine
        StringBuilder sb = new StringBuilder();
        for (String s : strings) {
            sb.append(s);
        }

        // remove leading 0s
        int index = 0;
        while (index < sb.length() && sb.charAt(index) == '0') {
            index++;
        }
        if (index == sb.length()) {
            return "0";
        } else {
            return sb.substring(index);
        }
    }

    /**
     * Largest Number.
     *
     * Given a list of non negative integers, arrange them such that they form
     * the largest number.
     *
     * Notice: The result may be very large, so you need to return a string
     * instead of an integer.
     *
     * Example: Given [1, 20, 23, 4, 8], the largest formed number is 8423201.
     *
     * Challenge: Do it in O(nlogn) time complexity.
     *
     * @param num:
     *            A list of non negative integers
     * @return: A string
     */
    @tags.Sort
    public String largestNumber(int[] num) {
        String[] nums = new String[num.length];
        for (int i = 0; i < num.length; i++) {
            nums[i] = String.valueOf(num[i]);
        }

        Arrays.sort(nums, new Comparator<String>() {
            @Override
            public int compare(String s1, String s2) {
                StringBuilder sb1 = new StringBuilder(s1);
                StringBuilder sb2 = new StringBuilder(s2);
                while (sb1.length() < sb2.length()) {
                    sb1.append(sb1.charAt(0));
                }
                while (sb2.length() < sb1.length()) {
                    sb2.append(sb2.charAt(0));
                }
                return sb2.toString().compareTo(sb1.toString());
            }
        });

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < nums.length; i++) {
            sb.append(nums[i]);
        }

        if (sb.charAt(0) == '0') {
            return "0";
        }
        return sb.toString();
    }

    /**
     * String to Integer (atoi).
     *
     * Implement function atoi to convert a string to an integer. If no valid
     * conversion could be performed, a zero value is returned. If the correct
     * value is out of the range of representable values, INT_MAX (2147483647)
     * or INT_MIN (-2147483648) is returned.
     *
     * Example: "123123123123123"=>2147483647, " 52abc "=>52, "10"=>10,
     * "1.0"=>1, "1234567890123456789012345678901234567890"=>2147483647,
     * null=>0, "-1"=>-1.
     *
     * @param str:
     *            A string
     * @return An integer
     */
    @tags.BasicImplementation
    @tags.String
    @tags.Math
    @tags.Company.Uber
    @tags.Status.Hard
    public int atoi(String str) {
        if (str == null) {
            return 0;
        }

        // trim white spaces
        str = str.trim();
        if (str.length() == 0) {
            return 0;
        }

        // sign
        boolean positive = str.charAt(0) == '+' || str.charAt(0) != '-';
        if (!positive || str.charAt(0) == '+') {
            str = str.substring(1);
        }

        // parse number
        long num = 0;
        for (int i = 0; i < str.length(); i++) {
            int digit = str.charAt(i) - '0';
            if (digit > 9 || digit < 0) {
                break;
            }
            num = num * 10 + digit;
            if (positive && num > Integer.MAX_VALUE) {
                return Integer.MAX_VALUE;
            } else if (!positive && -num < Integer.MIN_VALUE) {
                return Integer.MIN_VALUE;
            }
        }

        return positive ? (int) num : (int) -num;
    }

    /**
     * Valid Number - regular expression.
     * 
     * Validate if a given string is numeric.
     * 
     * Some examples: "0" => true " 0.1 " => true "abc" => false "1 a" => false
     * "2e10" => true
     * 
     * Note: It is intended for the problem statement to be ambiguous. You
     * should gather all requirements up front before implementing one.
     */
    @tags.Math
    @tags.String
    @tags.Company.LinkedIn
    @tags.Status.Hard
    public boolean isNumberRegex(String s) {
        return s.matches(
                "^\\s*[+-]?(\\d+|\\d*\\.\\d+|\\d+\\.\\d*)([eE][+-]?\\d+)?\\s*$");
    }

    /**
     * Valid Number.
     */
    @tags.Math
    @tags.String
    @tags.Company.LinkedIn
    @tags.Status.Hard
    public boolean isNumber(String s) {
        // white space
        s = s.trim();

        // avoid "3e" which is false
        if (s.length() > 0 && s.charAt(s.length() - 1) == 'e') {
            return false;
        }

        // split by "e" to get original string or 2 strings
        String[] t = s.split("e");
        if (t.length == 0 || t.length > 2) {
            return false;
        }

        // make sure both 2 sides of e are valid numbers
        boolean res = valid(t[0], false);
        if (t.length > 1) {
            res = res && valid(t[1], true);
        }

        return res;
    }

    private boolean valid(String s, boolean hasDot) {
        // avoid "1+", "+", "+."

        // remove sign
        if (s.length() > 0 && (s.charAt(0) == '+' || s.charAt(0) == '-')) {
            s = s.substring(1);
        }

        // empty or single dot
        if (s.length() == 0 || s.equals(".")) {
            return false;
        }

        // multiple dots and not 1 to 9
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '.') {
                if (hasDot)
                    return false;
                hasDot = true;
            } else if (!('0' <= s.charAt(i) && s.charAt(i) <= '9')) {
                return false;
            }
        }
        return true;
    }

    /**
     * Max Points on a Line.
     *
     * Given n points on a 2D plane, find the maximum number of points that lie
     * on the same straight line.
     *
     * Example: Given 4 points: (1,2), (3,6), (0,0), (1,3). The maximum number
     * is 3.
     *
     * @param points
     *            an array of point
     * @return an integer
     */
    @tags.Math
    @tags.HashTable
    @tags.Company.Apple
    @tags.Company.LinkedIn
    @tags.Company.Twitter
    @tags.Status.Hard
    public int maxPoints(Point[] points) {
        if (points == null || points.length == 0) {
            return 0;
        }

        int n = points.length;
        int max = 1;

        for (int i = 0; i < n; i++) {
            Map<Double, Integer> count = new HashMap<>(); // slop : count
            int dup = 1;

            for (int j = i + 1; j < n; j++) {
                // dup point
                if (points[i].x == points[j].x && points[i].y == points[j].y) {
                    dup++;
                    continue;
                }

                // slope is key
                // same x means vertical line
                // -0.0 != 0.0, -0.0 + 0.0 == 0
                double k = points[i].x == points[j].x ? Integer.MAX_VALUE
                        : 0.0 + (double) (points[i].y - points[j].y)
                                / (points[i].x - points[j].x);

                if (count.containsKey(k)) {
                    count.put(k, count.get(k) + 1);
                } else {
                    count.put(k, 1);
                }
            }

            // check new max
            for (Integer val : count.values()) {
                if (val + dup > max) {
                    max = val + dup;
                }
            }
            if (count.isEmpty()) {
                max = Math.max(max, dup);
            }
        }

        return max;
    }

    /**
     * Valid Perfect Square.
     *
     * Given a positive integer num, write a function which returns True if num
     * is a perfect square else False.
     *
     * Note: Do not use any built-in library function such as sqrt.
     *
     * Example 1: Input: 16. Returns: True.
     *
     * Example 2: Input: 14. Returns: False.
     *
     * @param num
     * @return
     */
    @tags.Math
    @tags.BinarySearch
    @tags.Company.LinkedIn
    public boolean isPerfectSquare(int num) {
        int start = 1, end = num;
        while (start <= end) {
            int mid = (start + end) >>> 1;
            long square = (long) mid * mid;
            if (square < num) {
                start = mid + 1;
            } else if (square > num) {
                end = mid - 1;
            } else {
                return true;
            }
        }

        return false;
    }

    /**
     * Sqrt(x) - O(logx) time.
     *
     * Implement int sqrt(int x). Compute and return the square root of x.
     *
     * Example: sqrt(3) = 1, sqrt(4) = 2, sqrt(5) = 2, sqrt(10) = 3.
     *
     * @param x:
     *            An integer
     * @return: The sqrt of x
     */
    @tags.BinarySearch
    @tags.Math
    @tags.Company.Apple
    @tags.Company.Bloomberg
    @tags.Company.Facebook
    public int sqrt(int x) {
        if (x < 0) {
            throw new IllegalArgumentException();
        }

        double error = 0.0000001f;
        double low = 0, high = x / 2 + 1;

        while (high - low > error) {
            double mid = (high + low) / 2;
            if (mid * mid > x) {
                high = mid;
            } else {
                low = mid;
            }
        }

        return (int) Math.floor(high);
    }

    /** Sqrt(x) - better solution. */
    @tags.BinarySearch
    @tags.Math
    @tags.Company.Facebook
    public int sqrt2(int x) {
        if (x < 0) {
            throw new IllegalArgumentException();
        }

        int start = 0, end = x;
        while (start < end) {
            int mid = (start + end + 1) >>> 1;
            if (mid > x / mid) {
                end = mid - 1;
            } else {
                start = mid;
            }
        }

        return end;
    }

    /**
     * Pow(x, n).
     *
     * Implement pow(x, n).
     *
     * Notice: You don't need to care about the precision of your answer, it's
     * acceptable if the expected answer and your answer 's difference is
     * smaller than 1e-3.
     *
     * Example: Pow(2.1, 3) = 9.261. Pow(0, 1) = 0. Pow(1, 0) = 1.
     *
     * Challenge: O(logn) time.
     *
     * @param x
     *            the base number
     * @param n
     *            the power number
     * @return the result
     */
    @tags.Math
    @tags.BinarySearch
    @tags.DivideAndConquer
    @tags.Company.Bloomberg
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Company.LinkedIn
    public double pow(double x, int n) {
        if (n == 0) {
            return 1;
        }

        double half = pow(x, n / 2);
        double result = half * half;

        if (n % 2 == 1) {
            result *= x;
        } else if (n % 2 == -1) {
            result /= x;
        }

        return result;
    }

    /**
     * Fast Power.
     *
     * Calculate the an % b where a, b and n are all 32bit integers.
     *
     * Example: For 231 % 3 = 2. For 1001000 % 1000 = 0.
     *
     * Challenge: O(logn).
     *
     * @param a,
     *            b, n: 32bit integers
     * @return: An integer
     */
    @tags.DivideAndConquer
    @tags.Status.Hard
    public int fastPower(int a, int b, int n) {
        if (n == 0) {
            return 1 % b;
        } else if (n == 1) {
            return a % b;
        }

        long half = fastPower(a, b, n / 2);
        long result = (half * half) % b;
        if (n % 2 == 1) {
            result = (result * a) % b;
        }

        return (int) result;
    }

    /**
     * Excel Sheet Column Title.
     *
     * Given a positive integer, return its corresponding column title as appear
     * in an Excel sheet.
     *
     * For example:
     * 1 -> A, 2 -> B, 3 -> C, ..., 26 -> Z, 27 -> AA, 28 -> AB.
     *
     * @param n
     * @return
     */
    @tags.Math
    @tags.Company.Facebook
    @tags.Company.Microsoft
    @tags.Company.Zenefits
    public String convertToTitle(int n) {
        int base = 26;

        if (n == 0) {
            return "A";
        }

        StringBuilder sb = new StringBuilder();

        while (n > 0) {
            n--;

            char c = (char) ('A' + n % base);
            sb.append(c);
            n /= base;
        }

        return sb.reverse().toString();
    }

    /**
     * Integer to English Words.
     *
     * Convert a non-negative integer to its english words representation. Given
     * input is guaranteed to be less than 231 - 1.
     *
     * For example,
     *
     * 123 -> "One Hundred Twenty Three"
     *
     * 12345 -> "Twelve Thousand Three Hundred Forty Five"
     *
     * 1234567 -> "One Million Two Hundred Thirty Four Thousand Five Hundred
     * Sixty Seven"
     *
     * Hint:
     *
     * Did you see a pattern in dividing the number into chunk of words? For
     * example, 123 and 123000.
     *
     * Group the number by thousands (3 digits). You can write a helper function
     * that takes a number less than 1000 and convert just that chunk to words.
     *
     * There are many edge cases. What are some good test cases? Does your code
     * work with input such as 0? Or 1000010? (middle chunk is zero and should
     * not be printed out)
     *
     * @param num
     * @return
     */
    @tags.Math
    @tags.String
    @tags.Company.Facebook
    @tags.Company.Microsoft
    public String numberToWords(int num) {
        if (num == 0) {
            return "Zero";
        }

        String[] units = { "", " Thousand", " Million", " Billion" };
        List<String> result = new ArrayList<>();

        for (String unit : units) {
            String number = subThousand(num % 1000);
            if (number.length() != 0) {
                result.add(number + unit);
            }
            num /= 1000;

            if (num == 0) {
                break;
            }
        }

        Collections.reverse(result);
        StringBuilder sb = new StringBuilder();
        for (String s : result) {
            sb.append(s);
            sb.append(' ');
        }
        return sb.substring(0, sb.length() - 1);
    }

    private String subThousand(int num) {
        String[] sub20 = { "", "One", "Two", "Three", "Four", "Five", "Six",
                "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen",
                "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen",
                "Nineteen" };
        String[] tens = { "", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty",
                "Seventy", "Eighty", "Ninety" };
        StringBuilder sb = new StringBuilder();

        // process hundred digit and clean it
        int hundred = num / 100;
        if (hundred != 0) {
            sb.append(sub20[hundred]);
            sb.append(" ");
            sb.append("Hundred");
        }
        num %= 100;

        // process ten digit (>= 20) and clean it
        int ten = num / 10;
        if (ten >= 2) {
            if (sb.length() != 0) {
                sb.append(" ");
            }
            sb.append(tens[ten]);
            num %= 10;
        }

        if (sb.length() != 0 && num != 0) {
            sb.append(" ");
        }
        sb.append(sub20[num]);

        return sb.toString();
    }

    // ---------------------------------------------------------------------- //
    // -------------------------- Bit Manipulation -------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * O(1) Check Power of 2 - same as below.
     *
     * Using O(1) time to check whether an integer n is a power of 2.
     *
     * Example: For n=4, return true; For n=5, return false;
     *
     * Challenge: O(1) time.
     *
     * @param n:
     *            An integer
     * @return: True or false
     */
    @tags.Math
    @tags.BitManipulation
    @tags.Company.Google
    @tags.Status.Hard
    public boolean checkPowerOf2(int n) {
        if (n <= 0) {
            return false;
        }
        return (n & (n - 1)) == 0;
    }

    /**
     * 231. Power of Two - same as above.
     *
     * Given an integer, write a function to determine if it is a power of two.
     *
     * @param n
     * @return
     */
    @tags.Math
    @tags.BitManipulation
    @tags.Source.LeetCode
    @tags.Company.Google
    @tags.Status.NeedPractice
    public boolean isPowerOfTwo(int n) {
        if (n < 1) {
            return false;
        }

        for (int num = 1; num <= n && num > 0; num <<= 1) {
            if (num == n) {
                return true;
            }
        }

        return false;
    }

    /**
     * A + B Problem.
     *
     * Write a function that add two numbers A and B. You should not use + or
     * any arithmetic operators.
     *
     * Notice: There is no need to read data from standard input stream. Both
     * parameters are given in function aplusb, you job is to calculate the sum
     * and return it.
     *
     * Clarification: Are a and b both 32-bit integers? Yes. Can I use bit
     * operation? Sure you can.
     *
     * Example: Given a=1 and b=2, return 3.
     *
     * Challenge: Of course you can just return a + b to get accepted. But Can
     * you challenge not do it like that?
     *
     * @param a:
     *            The first integer
     * @param b:
     *            The second integer
     * @return: The sum of a and b
     */
    @tags.BitManipulation
    @tags.Source.CrackingTheCodingInterview
    public int aplusb(int a, int b) {
        while (b != 0) {
            int mask = a & b;
            a = a | b;
            b = mask;
            a = a ^ b;
            b = b << 1;
        }
        return a;
    }

    /**
     * Count 1 in Binary.
     *
     * Count how many 1 in binary representation of a 32-bit integer.
     *
     * Example: Given 32, return 1. Given 5, return 2. Given 1023, return 9.
     *
     * Challenge: If the integer is n bits with m 1 bits. Can you do it in O(m)
     * time?
     *
     * @param num:
     *            an integer
     * @return: an integer, the number of ones in num
     */
    @tags.Binary
    @tags.BitManipulation
    @tags.Status.NeedPractice
    public int countOnes(int num) {
        int count = 0;
        while (num != 0) {
            if (num % 2 != 0) {
                count++;
            }
            num >>>= 1;
        }
        return count;
    }

    /**
     * Flip Bits.
     *
     * Determine the number of bits required to flip if you want to convert
     * integer n to integer m.
     *
     * Notice: Both n and m are 32-bit integers.
     *
     * Example: Given n = 31 (11111), m = 14 (01110), return 2.
     *
     * @param a,
     *            b: Two integer return: An integer
     */
    @tags.BitManipulation
    @tags.Source.CrackingTheCodingInterview
    public static int bitSwapRequired(int a, int b) {
        // write your code here
        int count = 0;
        for (int c = a ^ b; c != 0; c = c >>> 1) {
            count += c & 1;
        }
        return count;
    }

    /**
     * Find the Missing Number (Missing Number).
     *
     * Given an array contains N numbers of 0 .. N, find which number doesn't
     * exist in the array.
     *
     * Example: Given N = 3 and the array [0, 1, 3], return 2.
     *
     * Challenge: Do it in-place with O(1) extra memory and O(n) time.
     *
     * @param nums:
     *            an array of integers
     * @return: an integer
     */
    @tags.Array
    @tags.Math
    @tags.BitManipulation
    @tags.Greedy
    @tags.Company.Bloomberg
    @tags.Company.Microsoft
    public int findMissing(int[] nums) {
        int sum = 0, len = nums.length;
        for (int i = 0; i < len; i++) {
            sum += nums[i];
        }
        return len * (len + 1) / 2 - sum;
    }

    /**
     * Update Bits.
     *
     * Given two 32-bit numbers, N and M, and two bit positions, i and j. Write
     * a method to set all bits between i and j in N equal to M (e g , M becomes
     * a substring of N located at i and starting at j)
     *
     * Notice: In the function, the numbers N and M will given in decimal, you
     * should also return a decimal number.
     *
     * Clarification: You can assume that the bits j through i have enough space
     * to fit all of M. That is, if M=10011£¬ you can assume that there are at
     * least 5 bits between j and i. You would not, for example, have j=3 and
     * i=2, because M could not fully fit between bit 3 and bit 2.
     *
     * Example: Given N=(10000000000)2, M=(10101)2, i=2, j=6. return
     * N=(10001010100)2
     *
     * Challenge: Minimum number of operations?
     *
     * @param n,
     *            m: Two integer
     * @param i,
     *            j: Two bit positions return: An integer
     */
    @tags.BitManipulation
    @tags.Source.CrackingTheCodingInterview
    public int updateBits(int n, int m, int i, int j) {
        // get mask to clear i to j in n
        int mask = Integer.MIN_VALUE >> (j - i);
        mask >>>= (31 - j);
        mask = ~mask;
        n = n & mask;

        // get mask of m being i to j
        m = m << i;

        return m | n;
    }

    /**
     * Binary Representation.
     *
     * Given a (decimal - e.g. 3.72) number that is passed in as a string,
     * return the binary representation that is passed in as a string. If the
     * fractional part of the number can not be represented accurately in binary
     * with at most 32 characters, return ERROR.
     *
     * Example: For n = "3.72", return "ERROR". For n = "3.5", return "11.1".
     *
     * @param n:
     *            Given a decimal number that is passed in as a string
     * @return: A string
     */
    @tags.String
    @tags.BitManipulation
    @tags.Source.CrackingTheCodingInterview
    public String binaryRepresentation(String n) {
        int dot = n.indexOf('.');
        int integer = (dot == -1) ? Integer.valueOf(n)
                : Integer.valueOf(n.substring(0, dot));

        StringBuilder sb1 = new StringBuilder();
        while (integer > 0) {
            sb1.append(integer % 2);
            integer /= 2;
        }
        if (sb1.length() == 0) {
            sb1.append(0);
        }
        sb1.reverse();

        if (dot == -1) {
            return sb1.toString();
        }

        double fractional = Double.valueOf(n.substring(dot));
        if (fractional == 0) {
            return sb1.toString();
        }

        StringBuilder sb2 = new StringBuilder();
        double sum = 0;
        for (double d = 0.5; sb2.length() < 32 && sum < fractional; d /= 2) {
            if (fractional >= sum + d) {
                sb2.append(1);
                sum += d;
            } else {
                sb2.append(0);
            }
        }

        if (sum != fractional) {
            return "ERROR";
        } else {
            return sb1.toString() + "." + sb2.toString();
        }
    }

    /**
     * Find the Difference.
     *
     * Given two strings s and t which consist of only lowercase letters. String
     * t is generated by random shuffling string s and then add one more letter
     * at a random position. Find the letter that was added in t.
     *
     * Example: Input: s = "abcd", t = "abcde". Output: e.
     *
     * Explanation: 'e' is the letter that was added.
     *
     * @param s
     * @param t
     * @return
     */
    @tags.BitManipulation
    @tags.HashTable
    @tags.Company.Google
    public char findTheDifference(String s, String t) {
        // another solution: xor all characters in both the strings

        Map<Character, Integer> map = new HashMap<>();

        // count characters in s
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (map.containsKey(c)) {
                map.put(c, map.get(c) + 1);
            } else {
                map.put(c, 1);
            }
        }

        // compare map with characters in t
        for (int i = 0; i < t.length(); i++) {
            char c = t.charAt(i);
            if (!map.containsKey(c) || map.get(c) == 0) {
                return c;
            }
            map.put(c, map.get(c) - 1);
        }

        throw new IllegalArgumentException();
    }

    /**
     * Maximum XOR of Two Numbers in an Array.
     *
     * Given a list of numbers, a[0], a[1], a[2], ¡­ , a[N-1], where 0 <= a[i] <
     * 2^32. Find the maximum result of a[i] XOR a[j].
     *
     * Could you do this in O(n) runtime?
     *
     * Input: [3, 10, 5, 25, 2, 8], Output: 28
     *
     * @param nums
     * @return
     */
    @tags.Company.Google
    public int findMaximumXOR(int[] nums) {
        int max = 0, mask = 0;
        for (int i = 31; i >= 0; i--) {
            mask |= 1 << i;

            Set<Integer> set = new HashSet<>();
            for (Integer num : nums) {
                set.add(num & mask);
            }

            int tmp = max | (1 << i);
            for (Integer integer : set) {
                if (set.contains(tmp ^ integer)) {
                    max = tmp;
                    break;
                }
            }
        }

        return max;
    }

    // ---------------------------------------------------------------------- //
    // --------------------------- Single Number ---------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Single Number.
     *
     * Given an array of integers, every element appears twice except for one.
     * Find that single one. Time: O(n). Space: O(0).
     *
     * If there's no space constraint, Map should be a common solution.
     *
     * @param A
     *            : an integer array
     * @return : a integer
     */
    @tags.Greedy
    @tags.Status.Easy
    public int singleNumber(int[] A) {
        // Since A^B^A == B, xor every other element with first one of the A
        for (int i = 1; i < A.length; i++) {
            A[0] ^= A[i];
        }
        return A[0];
    }

    /**
     * Single Number II.
     *
     * Given an array of integers, every element appears three times except for
     * one. Find that single one. Time: O(n). Space: O(0).
     *
     * Example: Given [1,1,2,3,3,3,2,2,4,1], return 4.
     *
     * So tricky!!! Three bitmask variables.
     *
     * @param A
     *            : An integer array
     * @return : An integer
     */
    @tags.Greedy
    @tags.Status.Hard
    public int singleNumberII(int[] A) {
        int ones = 0;   // represent the ith bit has appear once
        int twos = 0;   // represent the ith bit has appear twice
        int threes = 0; // represent the ith bit has appear three times

        for (int i = 0; i < A.length; i++) {
            threes = (threes & ~A[i]) | (twos & A[i]);
            twos = (twos & ~A[i]) | (ones & A[i]);
            ones = (ones ^ A[i]) & ~(threes | twos);
        }

        return ones;
    }

    /** Another approach, just do this. */
    @tags.Greedy
    @tags.Status.Hard
    public int singleNumberII2(int[] A) {
        if (A == null || A.length == 0) {
            return -1;
        }
        int num = 0;
        for (int i = 0; i < 32; i++) {
            int count = 0;
            for (int j = 0; j < A.length; j++) {
                count += (A[j] >> i) & 1;
            }
            if (count % 3 == 1) {
                num |= (1 << i);
            }
        }
        return num;
    }

    /**
     * Single Number III.
     *
     * Given 2*n + 2 numbers, every numbers occurs twice except two, find them.
     *
     * Example: Given [1,2,2,3,4,4,5,3] return 1 and 5.
     *
     * Challenge: O(n) time, O(1) extra space.
     *
     * @param A
     *            : An integer array
     * @return : Two integers
     */
    @tags.Greedy
    @tags.Source.LintCode
    @tags.Status.Hard
    public List<Integer> singleNumberIII(int[] A) {
        int xor = 0;
        for (int i = 0; i < A.length; i++) {
            xor ^= A[i];
        }

        int lastDiffBit = xor & ~(xor - 1);
        int group1 = 0, group2 = 0;
        for (int i = 0; i < A.length; i++) {
            if ((A[i] & lastDiffBit) == 0) {
                group1 ^= A[i];
            } else {
                group2 ^= A[i];
            }
        }

        List<Integer> result = new ArrayList<>();
        result.add(group1);
        result.add(group2);
        return result;
    }

    @Test
    public void test() {
        sqrtTest();
        isNumberTest();
    }

    private void sqrtTest() {
        Assert.assertEquals(0, sqrt(0));
        Assert.assertEquals(1, sqrt(3));
        Assert.assertEquals(2, sqrt(4));
        Assert.assertEquals(256, sqrt(65536));
        Assert.assertEquals(31622, sqrt(999999999));
    }

    private void isNumberTest() {
        Assert.assertTrue(isNumber("0"));
        Assert.assertTrue(isNumber("1."));
    }
}
