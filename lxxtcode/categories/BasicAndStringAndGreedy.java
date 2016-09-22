package categories;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.Test;

/**
 * Essential skill testers, string problems and math.
 *
 * @author Guangcheng Lu
 */
public class BasicAndStringAndGreedy {

    // ---------------------------------------------------------------------- //
    // ------------------------------- PROBLEMS ----------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Fizz Buzz.
     *
     * Given number n. Print number from 1 to n. But: when number is divided by
     * 3, print "fizz". when number is divided by 5, print "buzz". when number
     * is divided by both 3 and 5, print "fizz buzz".
     *
     * Example: If n = 15, you should return: [ "1", "2", "fizz", "4", "buzz",
     * "fizz", "7", "8", "fizz", "buzz", "11", "fizz", "13", "14", "fizz buzz" ]
     *
     * @param n:
     *            As description.
     * @return: A list of strings.
     */
    @tags.Enumeration
    @tags.BasicImplementation
    @tags.Status.OK
    public ArrayList<String> fizzBuzz(int n) {
        ArrayList<String> result = new ArrayList<>();

        for (int i = 1; i <= n; i++) {
            if (i % 15 == 0) {
                result.add("fizz buzz");
            } else if (i % 3 == 0) {
                result.add("fizz");
            } else if (i % 5 == 0) {
                result.add("buzz");
            } else {
                result.add(String.valueOf(i));
            }
        }

        return result;
    }

    /**
     * Fibonacci - O(n) solution.
     *
     * Find the Nth number in Fibonacci sequence.
     *
     * A Fibonacci sequence is defined as: The first two numbers are 0 and 1.
     * The i th number is the sum of i-1 th number and i-2 th number. The first
     * ten numbers in Fibonacci sequence is: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34 ...
     *
     * Notice: The Nth fibonacci number won't exceed the max value of signed
     * 32-bit integer in the test cases.
     *
     * Example: Given 1, return 0. Given 2, return 1. Given 10, return 34.
     *
     * @param n:
     *            an integer
     * @return an integer f(n)
     */
    @tags.Enumeration
    @tags.Math
    @tags.NonRecursion
    @tags.Status.Easy
    public static int fib4(int n) {
        int a = 0;
        int b = 1;

        while (n-- > 0) {
            int temp = a + b;
            a = b;
            b = temp;
        }

        return a;
    }

    /** Fibonacci - best O(1) solution. */
    @tags.Enumeration
    @tags.Math
    @tags.NonRecursion
    @tags.Status.Hard
    public static int fib5(int n) {
        double a = Math.sqrt(5);

        return (int) (1 / a * Math.pow((a + 1) / 2, n)
                - 1 / a * Math.pow((1 - a) / 2, n));
    }

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
     * Count and Say.
     *
     * The count-and-say sequence is the sequence of integers beginning as
     * follows: 1, 11, 21, 1211, 111221, ...
     *
     * 1 is read off as "one 1" or 11. 11 is read off as "two 1s" or 21. 21 is
     * read off as "one 2, then one 1" or 1211.
     *
     * Given an integer n, generate the nth sequence.
     *
     * Note: The sequence of integers will be represented as a string.
     */
    @tags.String
    @tags.Company.Facebook
    public String countAndSay(int n) {
        StringBuilder current = new StringBuilder("1");

        while (--n > 0) {
            StringBuilder next = new StringBuilder();
            char num = current.charAt(0);
            int count = 1;

            for (int i = 1; i < current.length(); i++) {
                char c = current.charAt(i);
                if (c == num) {
                    count++;
                } else {
                    next.append(count).append(num);
                    num = c;
                    count = 1;
                }
            }

            next.append(count).append(num);
            current = next;
        }

        return current.toString();
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

    /**
     * Space Replacement.
     *
     * Write a method to replace all spaces in a string with %20. The string is
     * given in a characters array, you can assume it has enough space for
     * replacement and you are given the true length of the string. You code
     * should also return the new length of the string after replacement.
     *
     * Notice: If you are using Java or Python£¬please use characters array
     * instead of string.
     *
     * Example: Given "Mr John Smith", length = 13. The string after replacement
     * should be "Mr%20John%20Smith", you need to change the string in-place and
     * return the new length 17.
     *
     * Challenge: Do it in-place.
     *
     * @param string:
     *            An array of Char
     * @param length:
     *            The true length of the string
     * @return: The true length of new string
     */
    @tags.String
    @tags.Source.CrackingTheCodingInterview
    public int replaceBlank(char[] string, int length) {
        int newLen = 0;
        for (int i = 0; i < length; i++) {
            if (string[i] == ' ') {
                newLen += 3;
            } else {
                newLen++;
            }
        }

        for (int i = length - 1, len = newLen; i >= 0; i--) {
            if (string[i] == ' ') {
                string[--len] = '0';
                string[--len] = '2';
                string[--len] = '%';
            } else {
                string[--len] = string[i];
            }
        }

        return newLen;
    }

    /**
     * Perfect Rectangle.
     *
     * Given N axis-aligned rectangles where N > 0, determine if they all
     * together form an exact cover of a rectangular region.
     *
     * Each rectangle is represented as a bottom-left point and a top-right
     * point. For example, a unit square is represented as [1,1,2,2].
     * (coordinate of bottom-left point is (1, 1) and top-right point is (2,
     * 2)).
     *
     * @param rectangles
     * @return
     */
    @tags.Company.Google
    @tags.Status.Hard
    public boolean isRectangleCover(int[][] rectangles) {
        if (rectangles == null || rectangles.length <= 1) {
            return true;
        }

        int[] range = { Integer.MAX_VALUE, Integer.MAX_VALUE, 0, 0 };
        Set<String> set = new HashSet<>();
        int area = 0;

        for (int i = 0; i < rectangles.length; i++) {
            int left = rectangles[i][0], bottom = rectangles[i][1],
                    right = rectangles[i][2], top = rectangles[i][3];
            range[0] = Math.min(range[0], left);
            range[1] = Math.min(range[1], bottom);
            range[2] = Math.max(range[2], right);
            range[3] = Math.max(range[3], top);

            String bottomleft = String.valueOf(bottom) + " "
                    + String.valueOf(left);
            String bottomright = String.valueOf(bottom) + " "
                    + String.valueOf(right);
            String topleft = String.valueOf(top) + " " + String.valueOf(left);
            String topright = String.valueOf(top) + " " + String.valueOf(right);
            if (set.contains(bottomleft)) {
                set.remove(bottomleft);
            } else {
                set.add(bottomleft);
            }
            if (set.contains(bottomright)) {
                set.remove(bottomright);
            } else {
                set.add(bottomright);
            }
            if (set.contains(topleft)) {
                set.remove(topleft);
            } else {
                set.add(topleft);
            }
            if (set.contains(topright)) {
                set.remove(topright);
            } else {
                set.add(topright);
            }

            area += (rectangles[i][2] - rectangles[i][0])
                    * (rectangles[i][3] - rectangles[i][1]);
        }

        // make sure fully covered
        String bottomleft = String.valueOf(range[1]) + " "
                + String.valueOf(range[0]);
        String bottomright = String.valueOf(range[1]) + " "
                + String.valueOf(range[2]);
        String topleft = String.valueOf(range[3]) + " "
                + String.valueOf(range[0]);
        String topright = String.valueOf(range[3]) + " "
                + String.valueOf(range[2]);
        if (set.size() != 4 || !set.contains(bottomleft)
                || !set.contains(bottomright) || !set.contains(topleft)
                || !set.contains(topright)) {
            return false;
        }

        // make sure no overlap
        return area == (range[2] - range[0]) * (range[3] - range[1]);
    }

    /**
     * Digit Counts.
     *
     * Count the number of k's between 0 and n. k can be 0 - 9.
     *
     * Example: if n = 12, k = 1 in [0,1,2,3,4,5,6,7,8,9,10,11,12], we have FIVE
     * 1's (1,10,11,12).
     *
     * @param k
     *            As description. param n : As description.
     * @return An integer denote the count of digit k in 1..n.
     */
    @tags.Enumeration
    @tags.Status.OK
    public int digitCounts(int k, int n) {
        // another solution is not using string
        // count from k and consider edge cases

        char c = (char) k;
        c += '0';
        int count = 0;

        for (int num = 0; num <= n; num++) {
            String s = String.valueOf(num);
            for (int i = 0; i < s.length(); i++) {
                if (s.charAt(i) == c) {
                    count++;
                }
            }
        }

        return count;
    }

    /**
     * Create Maximum Number.
     *
     * Given two arrays of length m and n with digits 0-9 representing two
     * numbers. Create the maximum number of length k <= m + n from digits of
     * the two. The relative order of the digits from the same array must be
     * preserved. Return an array of the k digits. You should try to optimize
     * your time and space complexity.
     *
     * Example: Given nums1 = [3, 4, 6, 5], nums2 = [9, 1, 2, 5, 8, 3], k = 5,
     * return [9, 8, 6, 5, 3]. Given nums1 = [6, 7], nums2 = [6, 0, 4], k = 5,
     * return [6, 7, 6, 0, 4]. Given nums1 = [3, 9], nums2 = [8, 9], k = 3,
     * return [9, 8, 9].
     *
     * @param nums1
     *            an integer array of length m with digits 0-9
     * @param nums2
     *            an integer array of length n with digits 0-9
     * @param k
     *            an integer and k <= m + n
     * @return an integer array
     */
    @tags.Greedy
    @tags.DynamicProgramming
    @tags.Company.Google
    @tags.Status.Hard
    public int[] maxNumber(int[] nums1, int[] nums2, int k) {
        int m = nums1.length, n = nums2.length;
        if (m + n < k) {
            return null;
        } else if (m + n == k) {
            return merge(nums1, nums2);
        } else {
            int min = Math.max(0, k - n);
            int max = Math.min(k, m);
            int[] result = new int[k];

            for (int i = min; i <= max; i++) {
                int[] maxNumber = merge(getMax(nums1, i), getMax(nums2, k - i));
                if (isGreater(maxNumber, 0, result, 0)) {
                    result = maxNumber;
                }
            }
            return result;
        }
    }

    private int[] merge(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        if (m == 0) {
            return nums2;
        } else if (n == 0) {
            return nums1;
        }
        int[] result = new int[m + n];
        int i = 0, j = 0, pos = 0;
        while (pos < m + n) {
            if (isGreater(nums1, i, nums2, j)) {
                result[pos++] = nums1[i++];
            } else {
                result[pos++] = nums2[j++];
            }
        }

        return result;
    }

    private int[] getMax(int[] nums, int k) {
        if (k == 0) {
            return new int[0];
        }
        int[] result = new int[k];
        int index = 0;
        for (int i = 0; i < nums.length; i++) {
            while (index + nums.length - i > k && index > 0
                    && result[index - 1] < nums[i]) {
                index--;
            }
            if (index < k) {
                result[index++] = nums[i];
            }
        }
        return result;
    }

    private boolean isGreater(int[] nums1, int i1, int[] nums2, int i2) {
        for (; i1 < nums1.length && i2 < nums2.length; i1++, i2++) {
            if (nums1[i1] > nums2[i2]) {
                return true;
            } else if (nums1[i1] < nums2[i2]) {
                return false;
            }
        }
        return i1 != nums1.length;
    }

    /**
     * Rotate String.
     *
     * Given a string and an offset, rotate string by offset. (rotate from left
     * to right).
     *
     * Example: Given "abcdefg". offset=0 => "abcdefg". offset=1 => "gabcdef".
     * offset=2 => "fgabcde". offset=3 => "efgabcd".
     *
     * Challenge: Rotate in-place with O(1) extra memory.
     *
     * @param str:
     *            an array of char
     * @param offset:
     *            an integer
     * @return: nothing
     */
    @tags.String
    @tags.Status.NeedPractice
    public void rotateString(char[] str, int offset) {
        if (str != null && str.length > 0) {
            int len = str.length;
            offset %= len;
            while (offset-- > 0) {
                char end = str[str.length - 1];
                for (int i = str.length - 1; i > 0; i--) {
                    str[i] = str[i - 1];
                }
                str[0] = end;
            }
        }
    }

    /** Rotate String - tricky solution. */
    @tags.String
    @tags.Status.Hard
    public char[] rotateString2(char[] A, int offset) {
        if (A == null || A.length == 0) {
            return A;
        }

        offset = offset % A.length;
        reverse(A, 0, A.length - offset - 1);
        reverse(A, A.length - offset, A.length - 1);
        reverse(A, 0, A.length - 1);
        return A;
    }

    private void reverse(char[] A, int start, int end) {
        for (int i = start, j = end; i < j; i++, j--) {
            char temp = A[i];
            A[i] = A[j];
            A[j] = temp;
        }
    }

    /**
     * Text Justification.
     *
     * Given an array of words and a length L, format the text such that each
     * line has exactly L characters and is fully (left and right) justified.
     *
     * You should pack your words in a greedy approach; that is, pack as many
     * words as you can in each line. Pad extra spaces ' ' when necessary so
     * that each line has exactly L characters.
     *
     * Extra spaces between words should be distributed as evenly as possible.
     * If the number of spaces on a line do not divide evenly between words, the
     * empty slots on the left will be assigned more spaces than the slots on
     * the right.
     *
     * For the last line of text, it should be left justified and no extra space
     * is inserted between words.
     *
     * For example, words: ["This", "is", "an", "example", "of", "text",
     * "justification."]. L: 16.
     *
     * Return the formatted lines as: [ "This is an", "example of text",
     * "justification. " ].
     *
     * Note: Each word is guaranteed not to exceed L in length.
     *
     * Corner Cases: A line other than the last line might contain only one
     * word. What should you do in this case? In this case, that line should be
     * left-justified.
     */
    @tags.String
    @tags.Company.Airbnb
    @tags.Company.Facebook
    @tags.Company.LinkedIn
    @tags.Status.Hard
    public ArrayList<String> fullJustify(String[] words, int L) {
        ArrayList<String> result = new ArrayList<String>();
        if (words == null || words.length == 0) {
            return result;
        }

        int left = L - words[0].length();
        int start = 0;
        int end = 0;
        int spaceSlots = 0;

        for (int i = 1; i <= words.length; i++) {
            // can not add new word
            if (i == words.length || left - words[i].length() - (end - start + 1) < 0) {
                // construct new line
                StringBuffer line = new StringBuffer(L);
                if (spaceSlots == 0) {
                    line.append(words[start]);
                    for (int j = words[start].length(); j < L; j++) {
                        line.append(" ");
                    }
                } else {
                    for (int j = start; j < end; j++) {
                        line.append(words[j]);

                        if (i == words.length) {
                            line.append(" ");
                        } else {
                            int space = left / spaceSlots;
                            if (left % spaceSlots > 0) {
                                space++;
                            }

                            left -= space;

                            while (space-- > 0) {
                                line.append(" ");
                            }
                        }

                        spaceSlots--;
                    }
                    line.append(words[end]);
                    for (int j = line.length(); j < L; j++) {
                        line.append(" ");
                    }
                }
                result.add(line.toString());

                // start next round
                if (i < words.length) {
                    start = i;
                    end = i;
                    left = L - words[i].length();
                    spaceSlots = 0;
                }
            } else {
                left -= words[i].length();
                end = i;
                spaceSlots++;
            }
        }

        return result;
    }

    /** Other's solution */
    @tags.String
    @tags.Company.Airbnb
    @tags.Company.Facebook
    @tags.Company.LinkedIn
    @tags.Status.Hard
    public ArrayList<String> fullJustify2(String[] words, int L) {
        ArrayList<String> list = new ArrayList<String>();
        ArrayList<String> res = new ArrayList<String>();
        StringBuilder sb = new StringBuilder();
        int cur = 0;
        int len = 0;

        while (cur < words.length) {
            sb.setLength(0);
            sb.append(words[cur]);

            list.clear();
            len = words[cur].length();
            cur++;

            while (cur < words.length && len + 1 + words[cur].length() <= L) {
                list.add(" " + words[cur]);
                len += words[cur].length() + 1;

                cur++;

            }
            if (cur < words.length && list.size() > 0) {
                int spaces = L - len;
                int avg = spaces / list.size();
                int rem = spaces % list.size();
                for (int i = 0; i < list.size(); i++) {
                    appendSpaces(sb, i < rem ? avg + 1 : avg);
                    sb.append(list.get(i));
                }
            } else {
                for (int i = 0; i < list.size(); i++)
                    sb.append(list.get(i));
                // sb.append(list.get(0));
                appendSpaces(sb, L - len);
            }

            res.add(sb.toString());
        }
        return res;
    }

    private void appendSpaces(StringBuilder sb, int n) {
        for (int i = 0; i < n; i++)
            sb.append(' ');
    }

    /**
     * Repeated DNA Sequences.
     *
     * All DNA is composed of a series of nucleotides abbreviated as A, C, G,
     * and T, for example: "ACGAATTCCG". When studying DNA, it is sometimes
     * useful to identify repeated sequences within the DNA. Write a function to
     * find all the 10-letter-long sequences (substrings) that occur more than
     * once in a DNA molecule.
     *
     * For example, Given s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT",
     *
     * Return: ["AAAAACCCCC", "CCCCCAAAAA"].
     *
     * @param s
     * @return
     */
    @tags.HashTable
    @tags.BitManipulation
    @tags.Company.LinkedIn
    public List<String> findRepeatedDnaSequences(String s) {
        // TODO how to do bit manipulation?

        List<String> result = new ArrayList<>();
        if (s == null || s.length() == 0) {
            return result;
        }

        Map<String, Integer> count = new HashMap<>();
        for (int i = 0; i <= s.length() - 10; i++) {
            String seq = s.substring(i, i + 10);
            if (!count.containsKey(seq)) {
                count.put(seq, 0);
            }
            count.put(seq, count.get(seq) + 1);
        }

        for (String seq : count.keySet()) {
            if (count.get(seq) > 1) {
                result.add(seq);
            }
        }

        return result;
    }

    // ---------------------------------------------------------------------- //
    // ----------------------------- Flip Game ------------------------------ //
    // ---------------------------------------------------------------------- //

    /**
     * Flip Game.
     *
     * You are playing the following Flip Game with your friend: Given a string
     * that contains only these two characters: + and -, you and your friend
     * take turns to flip two consecutive "++" into "--". The game ends when a
     * person can no longer make a move and therefore the other person will be
     * the winner.
     *
     * Write a function to compute all possible states of the string after one
     * valid move.
     *
     * For example, given s = "++++", after one move, it may become one of the
     * following states:
     *
     * [ "--++", "+--+", "++--" ] If there is no valid move, return an empty
     * list [].
     *
     * @param s
     * @return
     */
    @tags.String
    @tags.Company.Google
    public List<String> generatePossibleNextMoves(String s) {
        List<String> result = new ArrayList<>();
        if (s == null || s.length() < 2) {
            return result;
        }

        StringBuilder sb = new StringBuilder(s);
        for (int i = 1; i < s.length(); i++) {
            if (s.charAt(i) == '+' && s.charAt(i - 1) == '+') {
                // flip
                sb.setCharAt(i, '-');
                sb.setCharAt(i - 1, '-');

                result.add(sb.toString());

                // back track
                sb.setCharAt(i, '+');
                sb.setCharAt(i - 1, '+');
            }
        }

        return result;
    }

    /**
     * Flip Game II.
     *
     * You are playing the following Flip Game with your friend: Given a string
     * that contains only these two characters: + and -, you and your friend
     * take turns to flip two consecutive "++" into "--". The game ends when a
     * person can no longer make a move and therefore the other person will be
     * the winner.
     *
     * Write a function to determine if the starting player can guarantee a win.
     *
     * For example, given s = "++++", return true. The starting player can
     * guarantee a win by flipping the middle "++" to become "+--+".
     *
     * Follow up: Derive your algorithm's runtime complexity.
     *
     * @param s
     * @return
     */
    @tags.Backtracking
    @tags.Company.Google
    public boolean canWin(String s) {
        if (finished(s)) {
            return false;
        }

        // Do one move first, and after all possible next moves by the other
        // player, current player has to make sure a win.
        //
        // Can use backtracking with StringBuilder to improve efficiency.
        List<String> oneMoves = generatePossibleNextMoves(s);
        for (String oneMove : oneMoves) {
            boolean canWin = true;
            for (String nextRound : generatePossibleNextMoves(oneMove)) {
                canWin = canWin && canWin(nextRound);
                if (!canWin) {
                    break;
                }

            }

            if (canWin) {
                return true;
            }
        }

        return false;
    }

    private boolean finished(String s) {
        if (s == null || s.length() < 2) {
            return true;
        }

        for (int i = 1; i < s.length(); i++) {
            if (s.charAt(i) == '+' && s.charAt(i - 1) == '+') {
                return false;
            }
        }

        return true;
    }

    // ---------------------------------------------------------------------- //
    // ---------------------------- Word Pattern ---------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Isomorphic Strings.
     *
     * Given two strings s and t, determine if they are isomorphic. Two strings
     * are isomorphic if the characters in s can be replaced to get t. All
     * occurrences of a character must be replaced with another character while
     * preserving the order of characters. No two characters may map to the same
     * character but a character may map to itself.
     *
     * For example, Given "egg", "add", return true. Given "foo", "bar", return
     * false. Given "paper", "title", return true.
     *
     * Note: You may assume both s and t have the same length.
     *
     * @param s
     * @param t
     * @return
     */
    @tags.String
    @tags.HashTable
    @tags.Company.LinkedIn
    @tags.Status.NeedPractice
    public boolean isIsomorphic(String s, String t) {
        if (s == null && t == null) {
            return true;
        } else if (s == null || t == null) {
            return false;
        }

        // different length
        int len = s.length();
        if (len != t.length()) {
            return false;
        }

        Map<Character, Character> map = new HashMap<>();
        char c = 'a';
        char[] ss = s.toCharArray();
        for (int i = 0; i < len; i++) {
            if (!map.containsKey(ss[i])) {
                map.put(ss[i], c++);
            }
            ss[i] = map.get(ss[i]);
        }

        map.clear();
        c = 'a';
        char[] tt = t.toCharArray();
        for (int i = 0; i < len; i++) {
            if (!map.containsKey(tt[i])) {
                map.put(tt[i], c++);
            }
            tt[i] = map.get(tt[i]);
        }

        return String.valueOf(ss).equals(String.valueOf(tt));
    }

    /** Isomorphic Strings - 2 maps solution. */
    @tags.String
    @tags.HashTable
    @tags.Company.LinkedIn
    @tags.Status.OK
    public boolean isIsomorphic2(String s, String t) {
        if (s == null || t == null) {
            return false;
        }

        Map<Character, Character> stMap = new HashMap<>();
        Map<Character, Character> tsMap = new HashMap<>();

        for (int i = 0; i < s.length(); i++) {
            char sc = s.charAt(i);
            char tc = t.charAt(i);

            if (!stMap.containsKey(sc) && !tsMap.containsKey(tc)) {
                stMap.put(sc, tc);
                tsMap.put(tc, sc);
                continue;
            }

            if (stMap.containsKey(sc) && tsMap.containsKey(tc)) {
                if (stMap.get(sc) == tc && tsMap.get(tc) == sc) {
                    continue;
                }
            }

            return false;
        }

        return true;
    }

    /**
     * Word Pattern.
     *
     * Given a pattern and a string str, find if str follows the same pattern.
     * Here follow means a full match, such that there is a bijection between a
     * letter in pattern and a non-empty word in str.
     *
     * Examples:
     * pattern = "abba", str = "dog cat cat dog" should return true.
     * pattern = "abba", str = "dog cat cat fish" should return false.
     * pattern = "aaaa", str = "dog cat cat dog" should return false.
     * pattern = "abba", str = "dog dog dog dog" should return false.
     *
     * Notes: You may assume
     * pattern contains only lowercase letters, and str contains lowercase
     * letters separated by a single space.
     *
     * @param pattern
     * @param str
     * @return
     */
    @tags.HashTable
    @tags.Company.Dropbox
    @tags.Company.Uber
    public boolean wordPattern(String pattern, String str) {
        String[] words = str.split(" ");
        if (words.length != pattern.length()) {
            return false;
        }

        Map<Character, String> map = new HashMap<>();
        Set<String> wordSet = new HashSet<>();

        for (int i = 0; i < pattern.length(); i++) {
            char c = pattern.charAt(i);

            if (!map.containsKey(c)) {
                map.put(c, words[i]);
            } else if (!map.get(c).equals(words[i])) {
                return false;
            }

            wordSet.add(words[i]);
        }

        return map.size() == wordSet.size();
    }

    /**
     * Word Pattern II.
     *
     * Given a pattern and a string str, find if str follows the same pattern.
     * Here follow means a full match, such that there is a bijection between a
     * letter in pattern and a non-empty substring in str.
     *
     * Examples:
     * pattern = "abab", str = "redblueredblue" should return true.
     * pattern = "aaaa", str = "asdasdasdasd" should return true.
     * pattern = "aabb", str = "xyzabcxzyabc" should return false.
     *
     * Notes:
     * You may assume both pattern and str contains only lowercase letters.
     *
     * @param pattern
     * @param str
     * @return
     */
    @tags.Backtracking
    @tags.Company.Dropbox
    @tags.Company.Uber
    public boolean wordPatternMatch(String pattern, String str) {
        if (pattern == null || str == null) {
            return false;
        }
        return wordPatternMatch(pattern, 0, str, 0,
                new HashMap<Character, String>(), new HashSet<String>());
    }

    private boolean wordPatternMatch(String pattern, int ip, String str, int is,
            Map<Character, String> map, Set<String> visited) {
        int pLen = pattern.length(), sLen = str.length();
        if (ip == pLen && is == sLen) {
            return true;
        } else if ((ip == pLen && is != sLen) || (ip != pLen && is == sLen)) {
            return false;
        }

        char c = pattern.charAt(ip);
        if (map.containsKey(c)) {
            String word = map.get(c);
            int len = word.length();
            if (is + len <= str.length()
                    && str.substring(is, is + len).equals(word)) {
                return wordPatternMatch(pattern, ip + 1, str, is + len, map,
                        visited);
            }
            return false;
        }

        for (int i = is; i < str.length(); i++) {
            String word = str.substring(is, i + 1);
            if (!visited.contains(word)) {
                map.put(c, word);
                visited.add(word);
                if (wordPatternMatch(pattern, ip + 1, str, i + 1, map,
                        visited)) {
                    return true;
                }
                visited.remove(word);
            }
        }
        map.remove(c);
        return false;
    }

    // ---------------------------------------------------------------------- //
    // ----------------------- Strobogrammatic Number ----------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Strobogrammatic Number.
     *
     * A strobogrammatic number is a number that looks the same when rotated 180
     * degrees (looked at upside down). Write a function to determine if a
     * number is strobogrammatic. The number is represented as a string.
     *
     * For example, the numbers "69", "88", and "818" are all strobogrammatic.
     *
     * @param num
     * @return
     */
    @tags.HashTable
    @tags.Math
    @tags.Company.Google
    @tags.Status.NeedPractice
    public boolean isStrobogrammatic(String num) {
        if (num == null) {
            return false;
        }

        Map<Character, Character> map = new HashMap<>();
        map.put('0', '0');
        map.put('1', '1');
        map.put('8', '8');
        map.put('6', '9');
        map.put('9', '6');

        for (int i = 0, j = num.length() - 1; i <= j; i++, j--) {
            if (!map.containsKey(num.charAt(i))
                    || map.get(num.charAt(i)) != num.charAt(j)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Strobogrammatic Number II.
     *
     * A strobogrammatic number is a number that looks the same when rotated 180
     * degrees (looked at upside down). Find all strobogrammatic numbers that
     * are of length = n.
     *
     * For example, Given n = 2, return ["11","69","88","96"].
     *
     * Hint: Try to use recursion and notice that it should recurse with n - 2
     * instead of n - 1.
     *
     * @param n
     * @return
     */
    @tags.Recursion
    @tags.Math
    @tags.Company.Google
    @tags.Status.NeedPractice
    public List<String> findStrobogrammatic(int n) {
        // Another cleaner recursive solution:
        // recursively wrapping from outer to inner with both n and current
        // layer, wrap with 0 only if not outermost layer

        List<String> result = new ArrayList<>();
        if (n == 0) {
            result.add("");
            return result;
        } else if (n == 1) {
            result.add("0");
            result.add("1");
            result.add("8");
            return result;
        }

        for (String s : findStrobogrammatic(n - 2)) {
            result.add("1" + s + "1");
            result.add("6" + s + "9");
            result.add("8" + s + "8");
            result.add("9" + s + "6");
            if (s.length() >= 2 && s.charAt(0) == '1') {
                s = s.substring(1, s.length() - 1);
                result.add("10" + s + "01");
                result.add("60" + s + "09");
                result.add("80" + s + "08");
                result.add("90" + s + "06");
            }
        }

        Collections.sort(result);
        return result;
    }

    /**
     * Strobogrammatic Number III.
     *
     * A strobogrammatic number is a number that looks the same when rotated 180
     * degrees (looked at upside down). Write a function to count the total
     * strobogrammatic numbers that exist in the range of low <= num <= high.
     *
     * For example, Given low = "50", high = "100", return 3. Because 69, 88,
     * and 96 are three strobogrammatic numbers.
     *
     * Note: Because the range might be a large number, the low and high numbers
     * are represented as string.
     *
     * @param low
     * @param high
     * @return
     */
    @tags.Math
    @tags.Recursion
    public int strobogrammaticInRange(String low, String high) {
        if (low == null || high == null) {
            return 0;
        }

        AtomicInteger count = new AtomicInteger(0);
        strobogrammaticInRange(low, high, "", count);
        strobogrammaticInRange(low, high, "0", count);
        strobogrammaticInRange(low, high, "1", count);
        strobogrammaticInRange(low, high, "8", count);
        return count.get();
    }

    private void strobogrammaticInRange(String low, String high, String path,
            AtomicInteger count) {
        // stop if larger than high
        if (gt(path, high)) {
            return;
        }

        // add if in range
        if (!gt(low, path) && (path.equals("0") || path.charAt(0) != '0')) {
            count.incrementAndGet();
        }

        // append
        if (path.length() < high.length() - 2) {
            strobogrammaticInRange(low, high, "0" + path + "0", count);
        }
        strobogrammaticInRange(low, high, "1" + path + "1", count);
        strobogrammaticInRange(low, high, "6" + path + "9", count);
        strobogrammaticInRange(low, high, "8" + path + "8", count);
        strobogrammaticInRange(low, high, "9" + path + "6", count);
    }

    private boolean gt(String n1, String n2) {
        if (n1.length() != n2.length()) {
            return n1.length() > n2.length();
        }
        return n1.compareTo(n2) > 0;
    }

    // ---------------------------------------------------------------------- //
    // -------------------------- Majority Number --------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Majority Number.
     *
     * Given an array of integers, the majority number is the number that occurs
     * more than half of the size of the array. Find it.
     *
     * Example: Given [1, 1, 1, 1, 2, 2, 2], return 1.
     *
     * Challenge: O(n) time and O(1) extra space.
     *
     * @param nums:
     *            a list of integers
     * @return: find a majority number
     */
    @tags.Greedy
    @tags.Enumeration
    @tags.Source.LintCode
    @tags.Company.Zenefits
    @tags.Status.OK
    public int majorityNumber(ArrayList<Integer> nums) {
        if (nums == null || nums.size() == 0) {
            throw new IllegalArgumentException();
        }

        int num = 0;
        int count = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (count == 0) {
                num = nums.get(i);
                count++;
            } else if (nums.get(i) == num) {
                count++;
            } else {
                count--;
            }
        }

        return num;
    }

    /**
     * Majority Number II.
     *
     * Given an array of integers, the majority number is the number that occurs
     * more than 1/3 of the size of the array. Find it.
     *
     * Notice: There is only one majority number in the array.
     *
     * Example: Given [1, 2, 1, 2, 1, 3, 3], return 1.
     *
     * Challenge: O(n) time and O(1) extra space.
     *
     * @param nums:
     *            A list of integers
     * @return: The majority number that occurs more than 1/3
     */
    @tags.Greedy
    @tags.Enumeration
    @tags.Source.LintCode
    @tags.Company.Zenefits
    @tags.Status.NeedPractice
    public int majorityNumberII(ArrayList<Integer> nums) {
        int candidate1 = 0, candidate2 = 0;
        int count1 = 0, count2 = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (nums.get(i) == candidate1) {
                count1++;
            } else if (nums.get(i) == candidate2) {
                count2++;
            } else if (count1 == 0) {
                candidate1 = nums.get(i);
                count1 = 1;
            } else if (count2 == 0) {
                candidate2 = nums.get(i);
                count2 = 1;
            } else {
                count1--;
                count2--;
            }
        }

        count1 = count2 = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (nums.get(i) == candidate1) {
                count1++;
            } else if (nums.get(i) == candidate2) {
                count2++;
            }
        }

        return count1 > count2 ? candidate1 : candidate2;
    }

    /**
     * Majority Number III.
     *
     * Given an array of integers and a number k, the majority number is the
     * number that occurs more than 1/k of the size of the array. Find it.
     *
     * Notice: There is only one majority number in the array.
     *
     * Example: Given [3,1,2,3,2,3,3,4,4,4] and k=3, return 3.
     *
     * Challenge: O(n) time and O(k) extra space.
     *
     * @param nums:
     *            A list of integers
     * @param k:
     *            As described
     * @return: The majority number
     */
    @tags.HashTable
    @tags.LinkedList
    @tags.Source.LintCode
    @tags.Status.NeedPractice
    public int majorityNumber(ArrayList<Integer> nums, int k) {
        Map<Integer, Integer> candidates = new HashMap<>();
        for (int i = 0; i < nums.size(); i++) {
            int num = nums.get(i);
            if (candidates.containsKey(num)) {
                candidates.put(num, candidates.get(num) + 1);
            } else if (candidates.size() < k) {
                candidates.put(num, 1);
            } else {
                Iterator<Integer> iter = candidates.keySet().iterator();
                while (iter.hasNext()) {
                    int candidate = iter.next();
                    int cnt = candidates.get(candidate);
                    if (cnt == 1) {
                        iter.remove();
                    } else {
                        candidates.put(candidate, cnt - 1);
                    }
                }
            }
        }

        for (Integer candidate : candidates.keySet()) {
            candidates.put(candidate, 0);
        }

        for (int i = 0; i < nums.size(); i++) {
            int num = nums.get(i);
            if (candidates.containsKey(num)) {
                candidates.put(num, candidates.get(num) + 1);
            }
        }

        int major = 0, count = 0;
        for (Integer candidate : candidates.keySet()) {
            if (candidates.get(candidate) > count) {
                major = candidate;
                count = candidates.get(candidate);
            }
        }

        return major;
    }

    // ---------------------------------------------------------------------- //
    // ---------------------------- Reverse Word ---------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Reverse Words in a String.
     *
     * Given an input string, reverse the string word by word.
     *
     * For example, Given s = "the sky is blue", return "blue is sky the".
     *
     * Clarification: What constitutes a word? A sequence of non-space
     * characters constitutes a word. Could the input string contain leading or
     * trailing spaces? Yes. However, your reversed string should not contain
     * leading or trailing spaces. How about multiple spaces between two words?
     * Reduce them to a single space in the reversed string.
     *
     * @param s : A string
     * @return : A string
     */
    @tags.String
    @tags.Status.NeedPractice
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

    /** Reverse Words in a String - another solution. */
    @tags.String
    @tags.Status.NeedPractice
    public String reverseWords2(String s) {
        String[] words = s.split(" +"); // note this regular expression
        if (words.length == 0) {
            return "";
        }

        for (int start = 0, end = words.length
                - 1; start < end; start++, end--) {
            String tmp = words[start];
            words[start] = words[end];
            words[end] = tmp;
        }

        // could be as simple as return String.join(" ", words) in Java 8
        StringBuilder sb = new StringBuilder();
        for (String word : words) {
            sb.append(word).append(' ');
        }

        return sb.deleteCharAt(sb.length() - 1).toString();
    }

    /**
     * Reverse Vowels of a String.
     *
     * Write a function that takes a string as input and reverse only the vowels
     * of a string.
     *
     * Example 1: Given s = "hello", return "holle". Example 2: Given s =
     * "leetcode", return "leotcede".
     *
     * Note: The vowels does not include the letter "y".
     *
     * @param s
     * @return
     */
    @tags.String
    @tags.TwoPointers
    @tags.Company.Google
    @tags.Status.OK
    public String reverseVowels(String s) {
        char[] letters = s.toCharArray();
        int start = 0, end = letters.length - 1;
        while (start < end) {
            if (!isVowel(letters[start])) {
                start++;
                continue;
            } else if (!isVowel(letters[end])) {
                end--;
                continue;
            }
            char c = letters[start];
            letters[start] = letters[end];
            letters[end] = c;
            start++;
            end--;
        }

        return String.valueOf(letters);
    }

    private boolean isVowel(char c) {
        char[] vowels = { 'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U' };
        for (int i = 0; i < vowels.length; i++) {
            if (vowels[i] == c) {
                return true;
            }
        }
        return false;
    }

    // ---------------------------------------------------------------------- //
    // -------------------------------- Greedy ------------------------------ //
    // ---------------------------------------------------------------------- //

    /**
     * Gas Station.
     *
     * There are N gas stations along a circular route, where the amount of gas
     * at station i is gas[i]. You have a car with an unlimited gas tank and it
     * costs cost[i] of gas to travel from station i to its next station (i+1).
     * You begin the journey with an empty tank at one of the gas stations.
     * Return the starting gas station's index if you can travel around the
     * circuit once, otherwise return -1.
     *
     * Notice: The solution is guaranteed to be unique.
     *
     * Example: Given 4 gas stations with gas[i]=[1,1,3,1], and the
     * cost[i]=[2,2,1,1]. The starting gas station's index is 2.
     *
     * Challenge: O(n) time and O(1) extra space.
     *
     * @param gas:
     *            an array of integers
     * @param cost:
     *            an array of integers
     * @return: an integer
     */
    @tags.Greedy
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int sum = 0, index = 0, total = 0;
        for (int i = 0; i < gas.length; i++) {
            sum += gas[i] - cost[i];
            total += gas[i] - cost[i];
            if (sum < 0) {
                sum = 0;
                index = i + 1;
            }
        }

        return total < 0 ? -1 : index;
    }

    /**
     * Candy.
     *
     * There are N children standing in a line. Each child is assigned a rating
     * value. You are giving candies to these children subjected to the
     * following requirements: Each child must have at least one candy. Children
     * with a higher rating get more candies than their neighbors. What is the
     * minimum candies you must give?
     *
     * Example: Given ratings = [1, 2], return 3. Given ratings = [1, 1, 1],
     * return 3. Given ratings = [1, 2, 2], return 4. ([1,2,1]).
     *
     * @param ratings
     *            Children's ratings
     * @return the minimum candies you must give
     */
    @tags.Array
    @tags.Greedy
    public int candy(int[] ratings) {
        if (ratings == null || ratings.length == 0) {
            return 0;
        }

        int n = ratings.length;
        int[] candys = new int[n];
        Arrays.fill(candys, 1);

        // handle ascending sequence
        for (int i = 1; i < n; i++) {
            if (ratings[i] > ratings[i - 1]) {
                candys[i] = candys[i - 1] + 1;
            }
        }

        // handle descending sequence
        for (int i = n - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1]) {
                candys[i] = Math.max(candys[i], candys[i + 1] + 1);
            }
        }

        // sum up
        int min = 0;
        for (int i = 0; i < n; i++) {
            min += candys[i];
        }
        return min;
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

    // ---------------------------------------------------------------------- //
    // -------------------------------- SUDOKU ------------------------------ //
    // ---------------------------------------------------------------------- //

    /**
     * Valid Sudoku.
     *
     * Determine if a Sudoku is valid, according to: Sudoku Puzzles - The Rules.
     *
     * The Sudoku board could be partially filled, where empty cells are filled
     * with the character '.'.
     *
     * Note: A valid Sudoku board (partially filled) is not necessarily
     * solvable. Only the filled cells need to be validated.
     *
     * @param board:
     *            the board
     * @return: wether the Sudoku is valid
     */
    @tags.Matrix
    @tags.HashTable
    @tags.Company.Apple
    @tags.Company.Snapchat
    @tags.Company.Uber
    @tags.Status.NeedPractice
    public boolean isValidSudoku(char[][] board) {
        boolean[] visited = new boolean[9];

        // row
        for (int i = 0; i < 9; i++) {
            Arrays.fill(visited, false);
            for (int j = 0; j < 9; j++) {
                if (!process(visited, board[i][j]))
                    return false;
            }
        }

        // col
        for (int i = 0; i < 9; i++) {
            Arrays.fill(visited, false);
            for (int j = 0; j < 9; j++) {
                if (!process(visited, board[j][i]))
                    return false;
            }
        }

        // sub matrix
        for (int i = 0; i < 9; i += 3) {
            for (int j = 0; j < 9; j += 3) {
                Arrays.fill(visited, false);
                for (int k = 0; k < 9; k++) {
                    if (!process(visited, board[i + k / 3][j + k % 3]))
                        return false;
                }
            }
        }
        return true;
    }

    private boolean process(boolean[] visited, char c) {
        if (c == '.') {
            return true;
        }

        int num = c - '0';
        if (num < 1 || num > 9 || visited[num - 1]) {
            return false;
        }

        visited[num - 1] = true;
        return true;
    }

    /**
     * Sudoku Solver.
     *
     * Write a program to solve a Sudoku puzzle by filling the empty cells.
     * Empty cells are indicated by the character '.'. You may assume that there
     * will be only one unique solution.
     */
    @tags.HashTable
    @tags.Backtracking
    @tags.Company.Snapchat
    @tags.Company.Uber
    @tags.Status.NeedPractice
    public void solveSudoku(char[][] board) {
        solve(board);
    }

    private boolean solve(char[][] board) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] == '.') {
                    for (int k = 1; k <= 9; k++) {
                        board[i][j] = (char) ('0' + k);
                        if (isValid(board, i, j) && solve(board)) {
                            return true;
                        }
                    }
                    board[i][j] = '.';
                    return false;
                }
            }
        }
        return true;
    }

    private boolean isValid(char[][] board, int row, int col) {
        for (int i = 0; i < 9; i++) {
            if (i != col && board[row][i] == board[row][col]) {
                return false;
            }
        }

        for (int i = 0; i < 9; i++) {
            if (i != row && board[i][col] == board[row][col]) {
                return false;
            }
        }

        int r = 3 * (row / 3);
        int c = 3 * (col / 3);
        for (int i = r; i < r + 3; i++) {
            for (int j = c; j < c + 3; j++) {
                if ((i != row || j != col) && board[i][j] == board[row][col]) {
                    return false;
                }
            }
        }
        return true;
    }

    // --------------------------- OLD ---------------------------

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

    @Test
    public void test() {
    }
}
