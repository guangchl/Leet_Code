package categories;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Assert;
import org.junit.Test;

/**
 * Essential skill testers, string problems and math.
 *
 * @author Guangcheng Lu
 */
public class BasicAndStringAndMath {

    // ---------------------------------------------------------------------- //
    // ------------------------------- PROBLEMS ----------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Implement strStr().
     *
     * Returns a pointer to the first occurrence of needle in haystack, or null
     * if needle is not part of haystack.
     */
    @tags.BasicImplementation
    @tags.String
    public String strStr(String src, String dest) {
        if (src == null || dest == null) {
            return null;
        }

        int i, j;
        for (i = 0; i <= src.length() - dest.length(); i++) {
            for (j = 0; j < dest.length(); j++) {
                if (src.charAt(i + j) != dest.charAt(j)) {
                    break;
                }
            }

            if (j == dest.length()) {
                return src.substring(i, i + j);
            }
        }

        return null;
    }

    /** KMP */
    public String kmp(String haystack, String needle) {
        int m = haystack.length();
        int n = needle.length();
        if (n == 0) {
            return haystack;
        } else if (m < n) {
            return null;
        }

        // construct cover of needle
        int[] cover = new int[n];
        int iter = 0;
        for (int i = 1; i < n; i++) {
            while (i < n && needle.charAt(i) == needle.charAt(iter)) {
                cover[i] = cover[i - 1] + 1;
                i++;
                iter++;
            }
            iter = 0;
        }

        int i = 0;
        int j = 0;
        while (i < m && j < n && m - i >= n - j) {
            if (haystack.charAt(i) != needle.charAt(j)) {
                if (j == 0) {
                    i += 1;
                } else {
                    j = cover[j - 1];
                }
            } else {
                i++;
                j++;
            }
        }

        return (j == n) ? haystack.substring(i - n) : null;
    }

    /**
     * Longest Common Prefix (LCP).
     *
     * Given k strings, find the longest common prefix (LCP).
     *
     * Example: For strings "ABCD", "ABEF" and "ACEF", the LCP is "A", For
     * strings "ABCDEFG", "ABCEFG" and "ABCEFA", the LCP is "ABC".
     *
     * @param strs:
     *            A list of strings
     * @return: The longest common prefix
     */
    @tags.Enumeration
    @tags.BasicImplementation
    @tags.String
    @tags.Source.LintCode
    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) {
            return "";
        }

        for (int i = 0; i < strs[0].length(); i++) {
            char c = strs[0].charAt(i);
            for (int j = 1; j < strs.length; j++) {
                if (i == strs[j].length() || strs[j].charAt(i) != c) {
                    return strs[0].substring(0, i);
                }
            }
        }

        return strs[0];
    }

    // ---------------------------------------------------------------------- //
    // --------------------------------- MATH ------------------------------- //
    // ---------------------------------------------------------------------- //

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
     * O(1) Check Power of 2.
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
    @tags.BitManipulation
    public boolean checkPowerOf2(int n) {
        if (n <= 0) {
            return false;
        }
        return (n & (n - 1)) == 0;
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
     * Unique Binary Search Trees.
     *
     * Given n, how many structurally unique BSTs (binary search trees) that
     * store values 1...n?
     *
     * Example: Given n = 3, there are a total of 5 unique BST's.
     *
     * @paramn n: An integer
     * @return: An integer
     */
    @tags.DynamicProgramming
    @tags.CatalanNumber
    public int numTrees(int n) {
        int[] treeCount = new int[n + 1];
        treeCount[0] = 1;

        for (int i = 1; i <= n; i++) { // fill dp array increasingly
            for (int j = 0; j <= i - 1; j++) { // root can be any node
                treeCount[i] += treeCount[j] * treeCount[i - j - 1];
            }
        }

        return treeCount[n];
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
    public int atoi(String str) {
        if (str == null) {
            return 0;
        }

        // trim spaces
        String num = str.trim();

        // check decimal point
        int decimalPoint = num.indexOf('.');
        if (decimalPoint != -1) {
            num = num.substring(0, decimalPoint);
        }

        if (num.length() == 0) {
            return 0;
        }

        // positive or negative
        boolean positive = true;
        int i = 0;
        if (num.charAt(0) == '-') {
            positive = false;
            i++;
        } else if (num.charAt(0) == '+') {
            i++;
        }

        // get the value
        long val = 0; // double can hold even longer number
        for (; i < num.length(); i++) {
            int digit = num.charAt(i) - '0';
            if (digit < 0 || digit > 9) {
                break;
            }
            if (positive) {
                if (val > Integer.MAX_VALUE) {
                    return Integer.MAX_VALUE;
                }
                val = val * 10 + digit;
            } else {
                if (val < Integer.MIN_VALUE) {
                    return Integer.MIN_VALUE;
                }
                val = val * 10 - digit;
            }
        }

        return (int) val;
    }

    /**
     * Sqrt(x) - O(logx) time.
     *
     * Implement int sqrt(int x). Compute and return the square root of x.
     *
     * Example: sqrt(3) = 1, sqrt(4) = 2, sqrt(5) = 2, sqrt(10) = 3.
     *
     * @param x: An integer
     * @return: The sqrt of x
     */
    @tags.BinarySearch
    @tags.Math
    @tags.Company.Apple
    @tags.Company.Bloomberg
    @tags.Company.Facebook
    public int sqrt(int x) {
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

    // ---------------------------------------------------------------------- //
    // ------------------------------- ANAGRAMS ----------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Two Strings Are Anagrams (Valid Anagram).
     *
     * Write a method anagram(s,t) to decide if two strings are anagrams or not.
     *
     * Clarification: What is Anagram? Two strings are anagram if they can be
     * the same after change the order of characters.
     *
     * Example: Given s = "abcd", t = "dcab", return true. Given s = "ab", t =
     * "ab", return true. Given s = "ab", t = "ac", return false.
     *
     * Challenge: O(n) time, O(1) extra space.
     *
     * Follow up: What if the inputs contain unicode characters? How would you
     * adapt your solution to such case?
     *
     * @param s:
     *            The first string
     * @param b:
     *            The second string
     * @return true or false
     */
    @tags.String
    @tags.HashTable
    @tags.Sort
    @tags.Source.CrackingTheCodingInterview
    @tags.Company.Amazon
    @tags.Company.Uber
    @tags.Company.Yelp
    public boolean anagram(String s, String t) {
        if (s == null && t == null) {
            return true;
        } else if ((s == null || t == null) || (s.length() != t.length())) {
            return false;
        }

        Map<Character, Integer> charCount = new HashMap<>();
        int len = s.length();

        for (int i = 0; i < len; i++) {
            char c = s.charAt(i);
            if (charCount.containsKey(c)) {
                charCount.put(c, charCount.get(c) + 1);
            } else {
                charCount.put(c, 1);
            }
        }

        for (int i = 0; i < len; i++) {
            char c = t.charAt(i);
            if (!charCount.containsKey(c) || charCount.get(c) == 0) {
                return false;
            } else {
                charCount.put(c, charCount.get(c) - 1);
            }
        }

        // no need to check non-zero count in the map since size was checked
        return true;
    }

    /**
     * Anagrams.
     *
     * Given an array of strings, return all groups of strings that are
     * anagrams.
     *
     * Note: All inputs will be in lower-case.
     *
     * Example: Given ["lint", "intl", "inlt", "code"], return ["lint", "inlt",
     * "intl"]. Given ["ab", "ba", "cd", "dc", "e"], return ["ab", "ba", "cd",
     * "dc"].
     *
     * @param strs:
     *            A list of strings
     * @return: A list of strings
     */
    @tags.String
    @tags.HashTable
    @tags.Company.Facebook
    @tags.Company.Uber
    public List<String> anagrams(String[] strs) {
        if (strs == null || strs.length == 0) {
            return Collections.emptyList();
        }

        Map<String, List<String>> groups = new HashMap<>();
        for (String s : strs) {
            int[] chars = new int[26];
            for (int i = 0; i < s.length(); i++) {
                chars[s.charAt(i) - 'a']++;
            }

            String hash = hash(chars);
            if (groups.containsKey(hash)) {
                groups.get(hash).add(s);
            } else {
                List<String> list = new ArrayList<>();
                list.add(s);
                groups.put(hash, list);
            }
        }

        List<String> result = new ArrayList<>();
        for (String s : groups.keySet()) {
            if (groups.get(s).size() > 1) {
                result.addAll(groups.get(s));
            }
        }
        return result;
    }

    private String hash(int[] chars) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < chars.length; i++) {
            if (chars[i] != 0) {
                sb.append('a' + i);
                sb.append(chars[i]);
            }
        }
        return sb.toString();
    }

    /**
     * Group Anagrams.
     *
     * Given an array of strings, group anagrams together.
     *
     * For example, given: ["eat", "tea", "tan", "ate", "nat", "bat"], Return: [
     * ["ate", "eat","tea"], ["nat","tan"], ["bat"] ]
     *
     * @param strs:
     *            A list of strings
     */
    @tags.String
    @tags.HashTable
    @tags.Company.Amazon
    @tags.Company.Bloomberg
    @tags.Company.Facebook
    @tags.Company.Uber
    @tags.Company.Yelp
    public List<List<String>> groupAnagrams(String[] strs) {
        if (strs == null || strs.length == 0) {
            return Collections.emptyList();
        }

        Map<String, List<String>> groups = new HashMap<>();
        for (String s : strs) {
            int[] chars = new int[26];
            for (int i = 0; i < s.length(); i++) {
                chars[s.charAt(i) - 'a']++;
            }

            String hash = hash(chars);
            if (groups.containsKey(hash)) {
                groups.get(hash).add(s);
            } else {
                List<String> list = new ArrayList<>();
                list.add(s);
                groups.put(hash, list);
            }
        }

        List<List<String>> result = new ArrayList<>();
        for (String s : groups.keySet()) {
            result.add(groups.get(s));
        }
        return result;
    }

    /**
     * Compare Strings.
     *
     * Compare two strings A and B, determine whether A contains all of the
     * characters in B. The characters in string A and B are all Upper Case
     * letters.
     *
     * Notice: The characters of B in A are not necessary continuous or ordered.
     *
     * Example: For A = "ABCD", B = "ACD", return true. For A = "ABCD", B =
     * "AABC", return false.
     *
     * @param A
     *            : A string includes Upper Case letters
     * @param B
     *            : A string includes Upper Case letter
     * @return : if string A contains all of the characters in B return true
     *         else return false
     */
    @tags.BasicImplementation
    @tags.String
    @tags.Source.LintCode
    public boolean compareStrings(String A, String B) {
        int[] chars = new int[26];
        for (int i = 0; i < A.length(); i++) {
            chars[A.charAt(i) - 'A']++;
        }
        for (int i = 0; i < B.length(); i++) {
            int index = B.charAt(i) - 'A';
            chars[index]--;
            if (chars[index] < 0) {
                return false;
            }
        }
        return true;
    }

    // --------------------------- OLD ---------------------------

    /**
     * Reverse Words in a String
     */
    public String reverseWords(String s) {
        if (s == null || s.length() == 0) {
            return "";
        }

        int index = s.length() - 1;
        StringBuffer sb = new StringBuffer();

        while (index >= 0) {
            int end = index;
            while (end >= 0 && s.charAt(end) == ' ') {
                end--;
            }
            if (end < 0) {
                break;
            }

            int start = end;
            while (start >= 0 && s.charAt(start) != ' ') {
                start--;
            }

            sb.append(s.substring(start + 1, end + 1));
            sb.append(' ');
            index = start - 1;
        }

        return (sb.length() > 0) ? sb.substring(0, sb.length() - 1) : "";
    }

    /**
     * Single Number
     * 
     * Given an array of integers, every element appears twice except for one.
     * Find that single one. Time: O(n). Space: O(0).
     * 
     * If there's no space constraint, Map should be a common solution
     */
    public int singleNumber(int[] A) {
        if (A == null || A.length == 0) {
            return -1;
        }

        // Since A^B^A == B, xor every other element with first one of the A
        for (int i = 1; i < A.length; i++) {
            A[0] ^= A[i];
        }
        return A[0];
    }

    /**
     * Single Number II
     * 
     * Given an array of integers, every element appears three times except for
     * one. Find that single one. Time: O(n). Space: O(0).
     * 
     * So tricky!!! Three bitmask variables.
     */
    public int singleNumber2(int[] A) {
        int ones = 0; // represent the ith bit has appear once
        int twos = 0; // represent the ith bit has appear twice
        int threes = 0; // represent the ith bit has appear three times

        for (int i = 0; i < A.length; i++) {
            threes = (threes & ~A[i]) | (twos & A[i]);
            twos = (twos & ~A[i]) | (ones & A[i]);
            ones = (ones ^ A[i]) & ~(threes | twos);
        }

        return ones;
        // Another solution
        // int ones = 0, twos = 0, threes = 0;
        // for (int i = 0; i < n; i++) {
        // twos |= ones & A[i];
        // ones ^= A[i];
        // threes = ones & twos;
        // ones &= ~threes;
        // twos &= ~threes;
        // }
        // return ones;
    }

    /** Another approach */
    public int singleNumber22(int[] A) {
        if (A == null || A.length == 0) {
            return -1;
        }
        int result = 0;
        int[] bits = new int[32];
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < A.length; j++) {
                bits[i] += A[j] >> i & 1;
                bits[i] %= 3;
            }

            result |= (bits[i] << i);
        }
        return result;
    }

    /**
     * Pow(x, n)
     * 
     * Implement pow(x, n).
     */
    public double pow(double x, int n) {
        if (n == 0) {
            return 1;
        } else if (n == 1) {
            return x;
        } else if (n == -1) {
            return 1 / x;
        }

        double u = pow(x, n / 2);
        double result = u * u;

        if (n % 2 == 1) {
            result *= x;
        } else if (n % 2 == -1) {
            result /= x;
        }

        return result;
    }

    /**
     * Divide Two Integers
     * 
     * Divide two integers without using multiplication, division and mod
     * operator.
     */
    public int divide(int dividend, int divisor) {
        boolean negative = dividend < 0 ^ divisor < 0;

        long a = Math.abs((long) dividend);
        long b = Math.abs((long) divisor);
        int ans = 0;

        while (a >= b) {
            int shift = 0;
            while ((b << shift) <= a) {
                shift++;
            }
            ans += 1 << (shift - 1);
            a = a - (b << (shift - 1));
        }

        return negative ? -ans : ans;
    }

    /**
     * Valid Number
     * 
     * Validate if a given string is numeric.
     * 
     * Some examples: "0" => true " 0.1 " => true "abc" => false "1 a" => false
     * "2e10" => true
     * 
     * Note: It is intended for the problem statement to be ambiguous. You
     * should gather all requirements up front before implementing one.
     */
    public boolean isNumberRegex(String s) {
        return s.matches(
                "^\\s*[+-]?(\\d+|\\d*\\.\\d+|\\d+\\.\\d*)([eE][+-]?\\d+)?\\s*$");
    }

    public boolean isNumber(String s) {
        s = s.trim();
        if (s.length() > 0 && s.charAt(s.length() - 1) == 'e')
            return false; // avoid "3e" which is false
        String[] t = s.split("e");
        if (t.length == 0 || t.length > 2)
            return false;
        boolean res = valid(t[0], false);
        if (t.length > 1)
            res = res && valid(t[1], true);
        return res;
    }

    private boolean valid(String s, boolean hasDot) {
        if (s.length() > 0 && (s.charAt(0) == '+' || s.charAt(0) == '-')) // avoid
                                                                          // "1+",
                                                                          // "+",
                                                                          // "+."
            s = s.substring(1);
        char[] arr = s.toCharArray();
        if (arr.length == 0 || s.equals("."))
            return false;
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == '.') {
                if (hasDot)
                    return false;
                hasDot = true;
            } else if (!('0' <= arr[i] && arr[i] <= '9')) {
                return false;
            }
        }
        return true;
    }

    @Test
    public void test() {
        sqrtTest();
    }

    private void sqrtTest() {
        Assert.assertEquals(0, sqrt(0));
        Assert.assertEquals(1, sqrt(3));
        Assert.assertEquals(2, sqrt(4));
        Assert.assertEquals(256, sqrt(65536));
        Assert.assertEquals(31622, sqrt(999999999));
    }

}
