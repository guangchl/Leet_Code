package categories;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import org.junit.Test;

public class Contest3 {

    // ---------------------------------------------------------------------- //
    // ------------------------------- MODELS ------------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * This is the interface that allows for creating nested lists. You should
     * not implement it, or speculate about its implementation
     */
    public interface NestedInteger {

        // @return true if this NestedInteger holds a single integer,
        // rather than a nested list.
        public boolean isInteger();

        // @return the single integer that this NestedInteger holds,
        // if it holds a single integer
        // Return null if this NestedInteger holds a nested list
        public Integer getInteger();

        // @return the nested list that this NestedInteger holds,
        // if it holds a nested list
        // Return null if this NestedInteger holds a single integer
        public List<NestedInteger> getList();
    }

    /** Definition of TreeNode */
    public class TreeNode {
        public int val;
        public TreeNode left, right;

        public TreeNode(int val) {
            this.val = val;
            this.left = this.right = null;
        }
    }

    // ---------------------------------------------------------------------- //
    // ------------------------------ PROBLEMS ------------------------------ //
    // ---------------------------------------------------------------------- //

    /**
     * Flatten List.
     *
     * Given a list, each element in the list can be a list or integer. flatten
     * it into a simply list with integers.
     *
     * Notice: If the element in the given list is a list, it can contain list
     * too.
     *
     * Example: Given [1,2,[1,2]], return [1,2,1,2]. Given [4,[3,[2,[1]]]],
     * return [4,3,2,1].
     *
     * Challenge: Do it in non-recursive.
     *
     * Use Queue/Stack for non-recursive solution.
     *
     * @param nestedList
     *            a list of NestedInteger
     * @return a list of integer
     */
    @tags.DFS
    @tags.BFS
    @tags.Recursion
    @tags.NonRecursion
    @tags.Source.LintCode
    public List<Integer> flatten(List<NestedInteger> nestedList) {
        List<Integer> result = new ArrayList<>();
        flatten(result, nestedList);
        return result;
    }

    private void flatten(List<Integer> result, List<NestedInteger> nestedList) {
        if (nestedList == null)
            return;

        for (NestedInteger ni : nestedList) {
            if (ni.isInteger()) {
                result.add(ni.getInteger());
            } else {
                flatten(result, ni.getList());
            }
        }
    }

    /**
     * Clone Binary Tree.
     *
     * For the given binary tree, return a deep copy of it.
     *
     * @param root:
     *            The root of binary tree
     * @return root of new tree
     */
    @tags.BinaryTree
    @tags.Recursion
    public TreeNode cloneTree(TreeNode root) {
        if (root == null) {
            return null;
        }

        TreeNode clone = new TreeNode(root.val);
        clone.left = cloneTree(root.left);
        clone.right = cloneTree(root.right);
        return clone;
    }

    /**
     * Dices Sum.
     *
     * Throw n dices, the sum of the dices' faces is S. Given n, find the all
     * possible value of S along with its probability.
     *
     * Example: Given n = 1, return [ [1, 0.17], [2, 0.17], [3, 0.17], [4,
     * 0.17], [5, 0.17], [6, 0.17]].
     *
     * Do the division at the end to avoid difference made by precision lost,
     * while use BigDecimal is another way.
     *
     * @param n
     *            an integer
     * @return a list of Map.Entry<sum, probability>
     */
    @tags.DynamicProgramming
    @tags.Math
    @tags.Probability
    public List<Map.Entry<Integer, Double>> dicesSum(int n) {
        // init
        double[] sumsCount = new double[6 * n];
        for (int i = 0; i < 6; i++) {
            sumsCount[i] = 1;
        }

        // roll dice n times
        for (int i = 1; i < n; i++) {
            for (int j = 6 * i - 1; j >= i - 1; j--) {
                for (int k = 1; k <= 6; k++) {
                    sumsCount[j + k] += sumsCount[j];
                }
                sumsCount[j] = 0;
            }
        }

        List<Map.Entry<Integer, Double>> result = new ArrayList<>();
        double total = Math.pow(6, n);
        for (int i = n - 1; i < sumsCount.length; i++) {
            result.add(
                    new AbstractMap.SimpleEntry<>(i + 1, sumsCount[i] / total));
        }
        return result;
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

    @Test
    public void test() {
        someTest();
    }

    private void someTest() {
    }
}
