package categories;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;

import org.junit.Assert;

/**
 * Data Structure
 *
 * @author Guangcheng Lu
 */
public class DataStructure {

    // ---------------------------------------------------------------------- //
    // ------------------------------- MODELS ------------------------------- //
    // ---------------------------------------------------------------------- //

    /** Definition for ListNode */
    public class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
            next = null;
        }
    }

    /** Definition of TreeNode. */
    public class TreeNode {
        public int val;
        public TreeNode left, right;

        public TreeNode(int val) {
            this.val = val;
            this.left = this.right = null;
        }
    }

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

    // ---------------------------------------------------------------------- //
    // ------------------------------- PROBLEMS ----------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Happy Number.
     *
     * Write an algorithm to determine if a number is happy. A happy number is a
     * number defined by the following process: Starting with any positive
     * integer, replace the number by the sum of the squares of its digits, and
     * repeat the process until the number equals 1 (where it will stay), or it
     * loops endlessly in a cycle which does not include 1. Those numbers for
     * which this process ends in 1 are happy numbers.
     *
     * Example: 19 is a happy number. 1^2 + 9^2 = 82, 8^2 + 2^2 = 68, 6^2 + 8^2
     * = 100, 1^2 + 0^2 + 0^2 = 1.
     *
     * @param n
     *            an integer
     * @return true if this is a happy number or false
     */
    @tags.HashTable
    @tags.Math
    @tags.Status.OK
    public boolean isHappy(int n) {
        if (n <= 0) {
            return false;
        }

        Set<Integer> visited = new HashSet<>();
        while (n != 1) {
            if (visited.contains(n)) {
                return false;
            }
            visited.add(n);

            int sum = 0;
            while (n != 0) {
                sum += (n % 10) * (n % 10);
                n /= 10;
            }
            n = sum;
        }

        return true;
    }

    /**
     * Hash Function.
     *
     * In data structure Hash, hash function is used to convert a string(or any
     * other type) into an integer smaller than hash size and bigger or equal to
     * zero. The objective of designing a hash function is to "hash" the key as
     * unreasonable as possible. A good hash function can avoid collision as
     * less as possible. A widely used hash function algorithm is using a magic
     * number 33, consider any string as a 33 based big integer like follow:
     * hashcode("abcd") = (ascii(a) * 333 + ascii(b) * 332 + ascii(c) *33 +
     * ascii(d)) % HASH_SIZE = (97* 333 + 98 * 332 + 99 * 33 +100) % HASH_SIZE =
     * 3595978 % HASH_SIZE. Here HASH_SIZE is the capacity of the hash table
     * (you can assume a hash table is like an array with index 0 ~
     * HASH_SIZE-1).
     *
     * Given a string as a key and the size of hash table, return the hash value
     * of this key.
     *
     * Clarification: For this problem, you are not necessary to design your own
     * hash algorithm or consider any collision issue, you just need to
     * implement the algorithm as described.
     *
     * Example For key="abcd" and size=100, return 78.
     *
     * @param key:
     *            A String you should hash
     * @param HASH_SIZE:
     *            An integer
     * @return an integer
     */
    @tags.HashTable
    public int hashCode(char[] key, int HASH_SIZE) {
        if (key == null || key.length == 0 || HASH_SIZE < 1) {
            return 0;
        }

        final int MAGIC_NUMBER = 33;
        long result = 0;
        for (int i = 0; i < key.length; i++) {
            result = (result * MAGIC_NUMBER + key[i]) % HASH_SIZE;
        }

        return (int) result;
    }

    /**
     * Rehashing.
     *
     * The size of the hash table is not determinate at the very beginning. If
     * the total size of keys is too large (e.g. size >= capacity / 10), we
     * should double the size of the hash table and rehash every keys. Say you
     * have a hash table looks like below:
     *
     * size=3, capacity=4 [null, 21 ¡ý 9 ¡ý null, 14 ¡ý null, null]
     *
     * The hash function is: int hashcode(int key, int capacity) { return key %
     * capacity; }
     *
     * here we have three numbers, 9, 14 and 21, where 21 and 9 share the same
     * position as they all have the same hashcode 1 (21 % 4 = 9 % 4 = 1). We
     * store them in the hash table by linked list.
     *
     * rehashing this hash table, double the capacity, you will get: size=3,
     * capacity=8
     *
     * index: 0 1 2 3 4 5 6 7 hash : [null, 9, null, null, null, 21, 14, null]
     *
     * Given the original hash table, return the new hash table after rehashing.
     *
     * Notice: For negative integer in hash table, the position can be
     * calculated as follow: C++/Java: if you directly calculate -4 % 3 you will
     * get -1. You can use function: a % b = (a % b + b) % b to make it is a non
     * negative integer. Python: you can directly use -1 % 3, you will get 2
     * automatically.
     *
     * Example: Given [null, 21->9->null, 14->null, null], return [null,
     * 9->null, null, null, null, 21->null, 14->null, null].
     *
     * @param hashTable:
     *            A list of The first node of linked list
     * @return: A list of The first node of linked list which have twice size
     */
    @tags.HashTable
    @tags.Source.LintCode
    public ListNode[] rehashing(ListNode[] hashTable) {
        int capacity = hashTable.length * 2;
        ListNode[] newHT = new ListNode[capacity];

        for (ListNode list : hashTable) {
            while (list != null) {
                int hashVal = (list.val + capacity) % capacity;
                ListNode node = new ListNode(list.val);

                if (newHT[hashVal] != null) {
                    ListNode current = newHT[hashVal];
                    while (current.next != null) {
                        current = current.next;
                    }
                    current.next = node;
                } else {
                    newHT[hashVal] = node;
                }

                list = list.next;
            }
        }
        return newHT;
    }

    /**
     * Stack Sorting.
     *
     * Sort a stack in ascending order (with biggest terms on top). You may use
     * at most one additional stack to hold items, but you may not copy the
     * elements into any other data structure (e.g. array).
     *
     * Challenge: O(n^2) time is acceptable.
     *
     * @param stack
     *            an integer stack
     * @return void
     */
    @tags.Stack
    public void stackSorting(Stack<Integer> stack) {
        Stack<Integer> minStack = new Stack<>();
        while (!stack.isEmpty()) {
            int current = stack.pop();
            while (!minStack.isEmpty() && minStack.peek() < current) {
                stack.push(minStack.pop());
            }
            minStack.push(current);
        }

        while (!minStack.isEmpty()) {
            stack.push(minStack.pop());
        }
    }

    /**
     * Heapify.
     *
     * Given an integer array, heapify it into a min-heap array.
     *
     * For a heap array A, A[0] is the root of heap, and for each A[i], A[i * 2
     * + 1] is the left child of A[i] and A[i * 2 + 2] is the right child of
     * A[i].
     *
     * Clarification:
     *
     * What is heap? Heap is a data structure, which usually have three methods:
     * push, pop and top. where "push" add a new element the heap, "pop" delete
     * the minimum/maximum element in the heap, "top" return the minimum/maximum
     * element.
     *
     * What is heapify? Convert an unordered integer array into a heap array. If
     * it is min-heap, for each element A[i], we will get A[i * 2 + 1] >= A[i]
     * and A[i * 2 + 2] >= A[i].
     *
     * What if there is a lot of solutions? Return any of them.
     *
     * Example: Given [3,2,1,4,5], return [1,2,3,4,5] or any legal heap array.
     *
     * Challenge: O(n) time complexity
     *
     * @param A:
     *            Given an integer array
     * @return: void
     */
    @tags.Heap
    @tags.Source.LintCode
    public void heapify(int[] A) {
        for (int i = A.length / 2 - 1; i >= 0; i--) {
            heapify(A, i);
        }
    }

    private void heapify(int[] A, int i) {
        int left = left(i), right = right(i);
        if (left >= A.length) {
            return;
        }

        // find the smaller child
        int less = right;
        if (right >= A.length || A[right] > A[left]) {
            less = left;
        }

        if (A[i] > A[less]) {
            swap(A, i, less);
            heapify(A, less);
        }
    }

    private int left(int i) {
        return i * 2 + 1;
    }

    private int right(int i) {
        return i * 2 + 2;
    }

    private void swap(int[] A, int i1, int i2) {
        int temp = A[i1];
        A[i1] = A[i2];
        A[i2] = temp;
    }

    /**
     * Longest Consecutive Sequence.
     *
     * Given an unsorted array of integers, find the length of the longest
     * consecutive elements sequence.
     *
     * Clarification Your algorithm should run in O(n) complexity.
     *
     * Example: Given [100, 4, 200, 1, 3, 2], The longest consecutive elements
     * sequence is [1, 2, 3, 4]. Return its length: 4.
     *
     * @param nums:
     *            A list of integers
     * @return an integer
     */
    @tags.Array
    @tags.HashTable
    @tags.Status.Hard
    public int longestConsecutive(int[] num) {
        if (num == null || num.length == 0) {
            return 0;
        }

        Set<Integer> set = new HashSet<>();
        for (Integer i : num) {
            set.add(i);
        }

        int max = 0;
        while (!set.isEmpty()) {
            int left = set.iterator().next(), right = left;
            set.remove(left);

            while (set.contains(left - 1)) {
                left--;
                set.remove(left);
            }
            while (set.contains(right + 1)) {
                right++;
                set.remove(right);
            }
            max = Math.max(max, right - left + 1);
        }

        return max;
    }

    /**
     * Animal Shelter.
     *
     * An animal shelter holds only dogs and cats, and operates on a strictly
     * "first in, first out" basis. People must adopt either the "oldest" (based
     * on arrival time) of all animals at the shelter, or they can select
     * whether they would prefer a dog or a cat (and will receive the oldest
     * animal of that type). They cannot select which specific animal they would
     * like. Create the data structures to maintain this system and implement
     * operations such as enqueue, dequeueAny, dequeueDog and dequeueCat.
     *
     * Example: int CAT = 0, int DOG = 1. enqueue("james", DOG); enqueue("tom",
     * DOG); enqueue("mimi", CAT); dequeueAny(); // should return "james"
     * dequeueCat(); // should return "mimi" dequeueDog(); // should return
     * "tom".
     *
     * Challenge: Can you do it with single Queue?
     */
    @tags.Queue
    @tags.LinkedList
    public class AnimalShelter {
        class Animal {
            String name;
            int type;

            public Animal(String name, int type) {
                this.name = name;
                this.type = type;
            }
        }

        private Queue<Animal> all = new LinkedList<>();
        private Queue<Animal> cats = new LinkedList<>();
        private Queue<Animal> dogs = new LinkedList<>();
        private final int CAT = 0;
        private final int DOG = 1;

        public AnimalShelter() {
        }

        /**
         * @param name a string
         * @param type an integer, 1 if Animal is dog or 0
         * @return void
         */
        void enqueue(String name, int type) {
            all.offer(new Animal(name, type));
        }

        public String dequeueAny() {
            if (!dogs.isEmpty()) {
                return dogs.poll().name;
            } else if (!cats.isEmpty()) {
                return cats.poll().name;
            } else if (!all.isEmpty()) {
                return all.poll().name;
            }
            return "";
        }

        public String dequeueDog() {
            if (!dogs.isEmpty()) {
                return dogs.poll().name;
            }
            while (!all.isEmpty()) {
                Animal a = all.poll();
                if (a.type == DOG) {
                    return a.name;
                }
                cats.offer(a);
            }
            return "";
        }

        public String dequeueCat() {
            if (!cats.isEmpty()) {
                return cats.poll().name;
            }
            while (!all.isEmpty()) {
                Animal a = all.poll();
                if (a.type == CAT) {
                    return a.name;
                }
                dogs.offer(a);
            }
            return "";
        }
    }

    /**
     * Longest Absolute File Path.
     *
     * Suppose we abstract our file system by a string in the following manner:
     * The string "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext" represents: The
     * directory dir contains an empty sub-directory subdir1 and a sub-directory
     * subdir2 containing a file file.ext. The string
     * "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext"
     * represents: The directory dir contains two sub-directories subdir1 and
     * subdir2. subdir1 contains a file file1.ext and an empty second-level
     * sub-directory subsubdir1. subdir2 contains a second-level sub-directory
     * subsubdir2 containing a file file2.ext. We are interested in finding the
     * longest (number of characters) absolute path to a file within our file
     * system.
     *
     * For example, in the second example above, the longest absolute path is
     * "dir/subdir2/subsubdir2/file2.ext", and its length is 32 (not including
     * the double quotes). Given a string representing the file system in the
     * above format, return the length of the longest absolute path to file in
     * the abstracted file system. If there is no file in the system, return 0.
     *
     * Note: The name of a file contains at least a . and an extension. The name
     * of a directory or sub-directory will not contain a .. Time complexity
     * required: O(n) where n is the size of the input string.
     *
     * Notice that a/aa/aaa/file1.txt is not the longest file path, if there is
     * another path aaaaaaaaaaaaaaaaaaaaa/sth.png.
     *
     * @param input
     * @return
     */
    @tags.Company.Google
    @tags.Status.Hard
    public int lengthLongestPath(String input) {
        Stack<Integer> stack = new Stack<>();
        String[] files = input.split("\n");
        int max = 0;

        for (String file : files) {
            // find the level of the dir or file
            int level = file.lastIndexOf('\t') + 1;

            // remove deeper path
            while (stack.size() > level) {
                stack.pop();
            }

            if (isFile(file, level)) {
                max = Math.max(max,
                        calculatePrefixLength(stack) + file.length() - level);
            } else {
                stack.push(file.length() - level);
            }
        }

        return max;
    }

    private boolean isFile(String file, int start) {
        for (int i = start; i < file.length(); i++) {
            if (file.charAt(i) == '.') {
                return true;
            }
        }
        return false;
    }

    private int calculatePrefixLength(Stack<Integer> stack) {
        int len = 0;
        for (Integer i : stack) {
            len += i;
            len++;
        }
        return len;
    }

    // ---------------------------------------------------------------------- //
    // ------------------------------ Iterator ------------------------------ //
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
     * Flatten Nested List Iterator.
     *
     * Given a nested list of integers, implement an iterator to flatten it.
     * Each element is either an integer, or a list -- whose elements may also
     * be integers or other lists.
     *
     * Example: Given the list [[1,1],2,[1,1]], By calling next repeatedly until
     * hasNext returns false, the order of elements returned by next should be:
     * [1,1,2,1,1]. Given the list [1,[4,[6]]], By calling next repeatedly until
     * hasNext returns false, the order of elements returned by next should be:
     * [1,4,6].
     *
     * Your NestedIterator object will be instantiated and called as such:
     * NestedIterator i = new NestedIterator(nestedList); while (i.hasNext())
     * v.add(i.next());
     *
     * This problem is easy without implementing remove().
     */
    @tags.Recursion
    @tags.Stack
    @tags.DataStructureDesign
    @tags.Company.Google
    @tags.Status.NeedPractice
    public class NestedIterator implements Iterator<Integer> {
        Stack<Iterator<NestedInteger>> stack = new Stack<>();
        Integer next = null;

        public NestedIterator(List<NestedInteger> nestedList) {
            stack.push(nestedList.iterator());
        }

        // @return {int} the next element in the iteration
        @Override
        public Integer next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            Integer result = next;
            next = null;
            return result;
        }

        // @return {boolean} true if the iteration has more element or false
        @Override
        public boolean hasNext() {
            while (!stack.isEmpty() && next == null) {
                if (stack.peek().hasNext()) {
                    NestedInteger ni = stack.peek().next();
                    if (ni.isInteger()) {
                        next = ni.getInteger();
                    } else {
                        stack.push(ni.getList().iterator());
                    }
                } else {
                    stack.pop();
                }
            }

            return next != null;
        }

        @Override
        public void remove() {
        }
    }

    /**
     * Nested List Weight Sum.
     *
     * Given a nested list of integers, return the sum of all integers in the
     * list weighted by their depth. Each element is either an integer, or a
     * list -- whose elements may also be integers or other lists.
     *
     * Example: Given the list [[1,1],2,[1,1]], return 10. (four 1's at depth 2,
     * one 2 at depth 1, 4 * 1 * 2 + 1 * 2 * 1 = 10) Given the list [1,[4,[6]]],
     * return 27. (one 1 at depth 1, one 4 at depth 2, and one 6 at depth 3; 1 +
     * 42 + 63 = 27)
     *
     * @param nestedList
     * @return
     */
    @tags.DFS
    @tags.Company.LinkedIn
    @tags.Status.OK
    public int depthSum(List<NestedInteger> nestedList) {
        return depthSum(nestedList, 1);
    }

    private int depthSum(List<NestedInteger> nestedList, int depth) {
        int sum = 0;
        for (int i = 0; i < nestedList.size(); i++) {
            NestedInteger ni = nestedList.get(i);
            if (ni.isInteger()) {
                sum += ni.getInteger() * depth;
            } else {
                sum += depthSum(ni.getList(), depth + 1);
            }
        }
        return sum;
    }

    /**
     * Nested List Weight Sum II.
     *
     * Given a nested list of integers, return the sum of all integers in the
     * list weighted by their depth. Each element is either an integer, or a
     * list -- whose elements may also be integers or other lists. Different
     * from the previous question where weight is increasing from root to leaf,
     * now the weight is defined from bottom up. i.e., the leaf level integers
     * have weight 1, and the root level integers have the largest weight.
     *
     * Example 1: Given the list [[1,1],2,[1,1]], return 8. (four 1's at depth
     * 1, one 2 at depth 2).
     *
     * Example 2: Given the list [1,[4,[6]]], return 17. (one 1 at depth 3, one
     * 4 at depth 2, and one 6 at depth 1; 1*3 + 4*2 + 6*1 = 17).
     */
    @tags.DFS
    @tags.Company.LinkedIn
    public int depthSumInverse(List<NestedInteger> nestedList) {
        // Very elegant solution, much better than calculate depth.
        // Another solution is calculate the unweighted as a list first, then
        // weight them.

        if (nestedList == null) {
            return 0;
        }

        int unweighted = 0;
        int weighted = 0;

        while (!nestedList.isEmpty()) {
            List<NestedInteger> nextLevel = new ArrayList<>();

            for (NestedInteger ni : nestedList) {
                if (ni.isInteger()) {
                    unweighted += ni.getInteger();
                } else {
                    nextLevel.addAll(ni.getList());
                }
            }

            // weighted is always aggregating the full result of unweighted
            weighted += unweighted;
            nestedList = nextLevel;
        }

        return weighted;
    }

    // ---------------------------------------------------------------------- //
    // -------------------------------- Stack ------------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Valid Parentheses
     *
     * Given a string containing just the characters '(', ')', '{', '}', '[' and
     * ']', determine if the input string is valid.
     *
     * The brackets must close in the correct order, "()" and "()[]{}" are all
     * valid but "(]" and "([)]" are not.
     */
    @tags.Stack
    @tags.String
    @tags.Company.Airbnb
    @tags.Company.Amazon
    @tags.Company.Bloomberg
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Company.Microsoft
    @tags.Company.Twitter
    @tags.Company.Zenefits
    @tags.Status.NeedPractice
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();

        for (int i = 0; i < s.length(); i++) {
            switch (s.charAt(i)) {
            case ')':
                if (stack.isEmpty() || stack.pop() != '(') {
                    return false;
                }
                break;
            case ']':
                if (stack.isEmpty() || stack.pop() != '[') {
                    return false;
                }
                break;
            case '}':
                if (stack.isEmpty() || stack.pop() != '{') {
                    return false;
                }
                break;
            default:
                stack.push(s.charAt(i));
                break;
            }
        }

        return stack.isEmpty();
    }

    /**
     * Longest Valid Parentheses.
     *
     * Given a string containing just the characters '(' and ')', find the
     * length of the longest valid (well-formed) parentheses substring.
     *
     * An example is ")()())", where the longest valid parentheses substring is
     * "()()", which has length = 4.
     *
     * DP would work, but this solution is much more elegant.
     */
    @tags.Stack
    @tags.DynamicProgramming
    @tags.String
    @tags.Status.Hard
    public int longestValidParentheses(String s) {
        Stack<Integer> stack = new Stack<>();
        int max = 0;

        for (int i = 0; i < s.length(); i++) {
            if (!stack.isEmpty() && s.charAt(stack.peek()) == '('
                    && s.charAt(i) == ')') {
                stack.pop();
                int lastEnd = stack.isEmpty() ? -1 : stack.peek();
                max = Math.max(max, i - lastEnd);
            } else {
                stack.push(i);
            }
        }

        return max;
    }

    /**
     * Remove Duplicate Letters.
     *
     * Given a string which contains only lowercase letters, remove duplicate
     * letters so that every letter appear once and only once. You must make
     * sure your result is the smallest in lexicographical order among all
     * possible results.
     *
     * Example: Given "bcabc", Return "abc". Given "cbacdcbc", Return "acdb".
     */
    @tags.Stack
    @tags.Greedy
    @tags.Company.Google
    @tags.Status.Hard
    public String removeDuplicateLetters(String s) {
        if (s == null || s.length() < 2) {
            return s;
        }

        // count letters
        Map<Character, Integer> count = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (count.containsKey(c)) {
                count.put(c, count.get(c) + 1);
            } else {
                count.put(c, 1);
            }
        }

        // find leftmost position each letter can be
        Set<Character> visited = new HashSet<>();
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (!visited.contains(c)) {
                while (!stack.isEmpty() && stack.peek() > c
                        && count.get(stack.peek()) > 0) {
                    visited.remove(stack.pop());
                }
                stack.push(c);
                visited.add(c);
            }
            count.put(c, count.get(c) - 1);
        }

        // reverse the stack to a string
        char[] string = new char[stack.size()];
        for (int i = string.length - 1; i >= 0; i--) {
            string[i] = stack.pop();
        }
        return String.valueOf(string);
    }

    // ---------------------------------------------------------------------- //
    // --------------------------- Priority Queue --------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Merge k Sorted Lists.
     *
     * Merge k sorted linked lists and return it as one sorted list. Analyze and
     * describe its complexity.
     *
     * @param lists:
     *            a list of ListNode
     * @return: The head of one sorted list.
     */
    @tags.DivideAndConquer
    @tags.LinkedList
    @tags.PriorityQueue
    @tags.Heap
    @tags.Company.Airbnb
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Company.LinkedIn
    @tags.Company.Twitter
    @tags.Company.Uber
    @tags.Status.Easy
    public ListNode mergeKLists(List<ListNode> lists) {
        if (lists == null || lists.size() == 0) {
            return null;
        }

        // construct min heap
        PriorityQueue<ListNode> minHeap = new PriorityQueue<ListNode>(
                lists.size(), new Comparator<ListNode>() {
                    @Override
                    public int compare(ListNode n1, ListNode n2) {
                        return n1.val - n2.val;
                    }
                });

        // insert head node of each list
        for (ListNode list : lists) {
            if (list != null) {
                minHeap.offer(list);
            }
        }

        // merge
        ListNode dummy = new ListNode(0);
        ListNode prev = dummy;
        while (!minHeap.isEmpty()) {
            prev.next = minHeap.poll();
            prev = prev.next;
            if (prev.next != null) {
                minHeap.offer(prev.next);
            }
        }

        return dummy.next;
    }

    /**
     * Merge k Sorted Arrays.
     *
     * Given k sorted integer arrays, merge them into one sorted array.
     *
     * Example: Given 3 sorted arrays: [ [1, 3, 5, 7], [2, 4, 6], [0, 8, 9, 10,
     * 11] ]. return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].
     *
     * Time complexity: O(N log k). N is the total number of integers, k is the
     * number of arrays.
     *
     * @param arrays
     *            k sorted integer arrays
     * @return a sorted array
     */
    @tags.PriorityQueue
    @tags.Heap
    public List<Integer> mergekSortedArrays(int[][] arrays) {
        if (arrays == null || arrays.length == 0) {
            return Collections.emptyList();
        }

        int m = arrays.length;

        class Element {
            int row;
            int column;
            int val;

            Element(int row, int column, int val) {
                this.row = row;
                this.column = column;
                this.val = val;
            }
        }

        PriorityQueue<Element> pq = new PriorityQueue<>(m,
                new Comparator<Element>() {
                    @Override
                    public int compare(Element e1, Element e2) {
                        return e1.val - e2.val;
                    }
                });

        for (int i = 0; i < m; i++) {
            if (arrays[i] != null && arrays[i].length != 0) {
                Element e = new Element(i, 0, arrays[i][0]);
                pq.offer(e);
            }
        }

        List<Integer> sorted = new ArrayList<>();
        while (!pq.isEmpty()) {
            Element e = pq.poll();
            sorted.add(e.val);
            int column = e.column + 1;
            if (column < arrays[e.row].length) {
                pq.offer(new Element(e.row, column, arrays[e.row][column]));
            }
        }

        return sorted;
    }

    /**
     * Top K Frequent Elements.
     *
     * Given a non-empty array of integers, return the k most frequent elements.
     *
     * For example, Given [1,1,1,2,2,3] and k = 2, return [1,2].
     *
     * Note: You may assume k is always valid, 1 ¡Ü k ¡Ü number of unique
     * elements. Your algorithm's time complexity must be better than O(n log
     * n), where n is the array's size.
     *
     * @param nums
     * @param k
     * @return
     */
    @tags.Heap
    @tags.PriorityQueue
    @tags.HashTable
    @tags.Company.LinkedIn
    @tags.Company.PocketGems
    @tags.Company.Yelp
    public List<Integer> topKFrequent(int[] nums, int k) {
        // count frequency
        final Map<Integer, Integer> count = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (!count.containsKey(nums[i])) {
                count.put(nums[i], 1);
            } else {
                count.put(nums[i], count.get(nums[i]) + 1);
            }
        }

        // min heap with k elements (nlogk)
        PriorityQueue<Integer> minHeap = new PriorityQueue<>(k + 1,
                new Comparator<Integer>() {
                    @Override
                    public int compare(Integer i1, Integer i2) {
                        return count.get(i1) - count.get(i2);
                    }
                });

        // compare all numbers' frequency
        for (Integer num : count.keySet()) {
            minHeap.offer(num);
            if (minHeap.size() == k + 1) {
                minHeap.poll();
            }
        }

        // add remaining elements in result
        List<Integer> topK = new ArrayList<>(k);
        while (!minHeap.isEmpty()) {
            topK.add(minHeap.poll());
        }

        return topK;
    }

    /**
     * Data Stream Median.
     *
     * Numbers keep coming, return the median of numbers at every time a new
     * number added.
     *
     * Example: For numbers coming list: [1, 2, 3, 4, 5], return [1, 1, 2, 2,
     * 3]. For numbers coming list: [4, 5, 1, 3, 2, 6, 0], return [4, 4, 4, 3,
     * 3, 3, 3]. For numbers coming list: [2, 20, 100], return [2, 2, 20].
     *
     * Challenge: Total run time in O(nlogn).
     *
     * @param nums:
     *            A list of integers.
     * @return: the median of numbers
     */
    @tags.Heap
    @tags.PriorityQueue
    @tags.Source.LintCode
    @tags.Company.Google
    @tags.Status.NeedPractice
    public int[] medianII(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new int[0];
        }

        int n = nums.length;
        int[] medians = new int[n];
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((n + 1) / 2,
                Collections.reverseOrder());
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();

        for (int i = 0; i < n; i++) {
            if (maxHeap.size() == minHeap.size()) {
                minHeap.offer(nums[i]);
                maxHeap.offer(minHeap.poll());
            } else {
                maxHeap.offer(nums[i]);
                minHeap.offer(maxHeap.poll());
            }

            medians[i] = maxHeap.peek();
        }

        return medians;
    }

    /**
     * Find Median from Data Stream - same as above.
     *
     * Median is the middle value in an ordered integer list. If the size of the
     * list is even, there is no middle value. So the median is the mean of the
     * two middle value.
     *
     * Examples: [2,3,4] , the median is 3. [2,3], the median is (2 + 3) / 2 =
     * 2.5.
     *
     * Design a data structure that supports the following two operations: void
     * addNum(int num) - Add a integer number from the data stream to the data
     * structure. double findMedian() - Return the median of all elements so
     * far.
     *
     * For example: add(1), add(2), findMedian() -> 1.5, add(3), findMedian() ->
     * 2.
     */
    @tags.Heap
    @tags.Design
    @tags.Company.Google
    @tags.Status.NeedPractice
    public class MedianFinder {
        // Your MedianFinder object will be instantiated and called as such:
        // MedianFinder mf = new MedianFinder();
        // mf.addNum(1);
        // mf.findMedian();

        private PriorityQueue<Integer> small = new PriorityQueue<>(1,
                Collections.reverseOrder());
        private PriorityQueue<Integer> large = new PriorityQueue<>();

        // Adds a number into the data structure.
        public void addNum(int num) {
            if (small.size() == large.size()) {
                large.offer(num);
                small.offer(large.poll());
            } else {
                small.offer(num);
                large.offer(small.poll());
            }
        }

        // Returns the median of current data stream
        public double findMedian() {
            return small.size() > large.size() ? (double) small.peek()
                    : ((double) small.peek() + large.peek()) / 2;
        }
    }

    /**
     * Sliding Window Median.
     *
     * Given an array of n integer, and a moving window(size k), move the window
     * at each iteration from the start of the array, find the median of the
     * element inside the window at each moving. (If there are even numbers in
     * the array, return the N/2-th number after sorting the element in the
     * window. )
     *
     * Example: For array [1,2,7,8,5], moving window size k = 3. return [2,7,7].
     * At first the window is at the start of the array like this: [ | 1,2,7 |
     * ,8,5] , return the median 2; then the window move one step forward. [1, |
     * 2,7,8 | ,5], return the median 7; then the window move one step forward
     * again. [1,2, | 7,8,5 | ], return the median 7;
     *
     * Challenge: O(nlog(n)) time.
     *
     * @param nums:
     *            A list of integers.
     * @return: The median of the element inside the window at each moving.
     */
    @tags.Heap
    @tags.PriorityQueue
    @tags.Source.LintCode
    @tags.Status.NeedPractice
    public ArrayList<Integer> medianSlidingWindow(int[] nums, int k) {
        ArrayList<Integer> medians = new ArrayList<>();
        if (nums == null || nums.length == 0 || nums.length < k) {
            return medians;
        }

        int n = nums.length;
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((n + 1) / 2,
                Collections.reverseOrder());
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();

        for (int i = 0; i < n; i++) {
            if (maxHeap.size() + minHeap.size() == k) {
                if (!minHeap.remove(nums[i - k])) {
                    maxHeap.remove(nums[i - k]);
                }
            }
            if (maxHeap.size() <= minHeap.size()) {
                minHeap.offer(nums[i]);
                maxHeap.offer(minHeap.poll());
            } else {
                maxHeap.offer(nums[i]);
                minHeap.offer(maxHeap.poll());
            }

            if (maxHeap.size() + minHeap.size() == k) {
                medians.add(maxHeap.peek());
            }
        }

        return medians;
    }

    /**
     * Sliding Window Maximum.
     *
     * Given an array of n integer with duplicate number, and a moving
     * window(size k), move the window at each iteration from the start of the
     * array, find the maximum number inside the window at each moving.
     *
     * Example: For array [1,2,7,7,8], moving window size k = 3. return [7,7,8].
     *
     * At first the window is at the start of the array like this,
     * [|1,2,7|,7,8], return the maximum 7; then the window move one step
     * forward.
     *
     * [1,|2,7,7|,8], return the maximum 7; then the window move one step
     * forward again.
     *
     * [1,2,|7,7,8|], return the maximum 8;
     *
     * Challenge: o(n) time and O(k) memory.
     *
     * @param nums:
     *            A list of integers.
     * @return: The maximum number inside the window at each moving.
     */
    @tags.Deque
    @tags.Company.Zenefits
    @tags.Source.LintCode
    @tags.Status.Hard
    public ArrayList<Integer> maxSlidingWindow(int[] nums, int k) {
        ArrayList<Integer> max = new ArrayList<>();
        Deque<Integer> deque = new LinkedList<>();

        for (int i = 0; i < nums.length; i++) {
            // first is the max
            // insert new number next to the larger one (keep descending order)

            while (!deque.isEmpty() && nums[i] > deque.peekLast()) {
                deque.pollLast();
            }
            deque.offerLast(nums[i]);

            if (i >= k && deque.peekFirst() == nums[i - k]) {
                deque.pollFirst();
            }

            if (i >= k - 1) {
                max.add(deque.peekFirst());
            }
        }

        return max;
    }

    /**
     * Moving Average from Data Stream.
     *
     * Given a stream of integers and a window size, calculate the moving
     * average of all integers in the sliding window.
     *
     * For example, MovingAverage m = new MovingAverage(3); m.next(1) = 1,
     * m.next(10) = (1 + 10) / 2, m.next(3) = (1 + 10 + 3) / 3, m.next(5) = (10
     * + 3 + 5) / 3.
     *
     */
    @tags.Design
    @tags.Queue
    @tags.Company.Google
    public class MovingAverage {
        // Your MovingAverage object will be instantiated and called as such:
        // MovingAverage obj = new MovingAverage(size);
        // double param_1 = obj.next(val);

        Queue<Integer> queue = new LinkedList<>();
        int sum, capacity;

        /** Initialize your data structure here. */
        public MovingAverage(int size) {
            capacity = size;
        }

        public double next(int val) {
            if (queue.size() == capacity) {
                sum -= queue.poll();
            }
            queue.offer(val);
            sum += val;
            return (double) sum / queue.size();
        }
    }

    // ---------------------------------------------------------------------- //
    // ----------------------- Implement Queue/Stack ------------------------ //
    // ---------------------------------------------------------------------- //

    /**
     * Implement Queue by Two Stacks.
     *
     * As the title described, you should only use two stacks to implement a
     * queue's actions. The queue should support push(element), pop() and top()
     * where pop is pop the first(a.k.a front) element in the queue. Both pop
     * and top methods should return the value of first element.
     *
     * Example: push(1), pop() return 1, push(2), push(3), top() return 2, pop()
     * return 2
     *
     * Challenge: implement it by two stacks, do not use any other data
     * structure and push, pop and top should be O(1) by AVERAGE.
     */
    @tags.Stack
    @tags.Queue
    @tags.Status.OK
    public class QueueTwoStack {
        private Stack<Integer> stack1;
        private Stack<Integer> stack2;

        public QueueTwoStack() {
            stack1 = new Stack<>();
            stack2 = new Stack<>();
        }

        public void push(int element) {
            stack1.push(element);
        }

        public int pop() {
            rollFrom1To2();
            return stack2.pop();
        }

        public int top() {
            rollFrom1To2();
            return stack2.peek();
        }

        private void rollFrom1To2() {
            if (stack2.isEmpty()) {
                while (!stack1.isEmpty()) {
                    stack2.push(stack1.pop());
                }
            }
        }
    }

    /**
     * Implement Stack.
     *
     * Implement a stack. You can use any data structure inside a stack except
     * stack itself to implement it.
     */
    @tags.Stack
    @tags.LinkedList
    @tags.Array
    class StackNoStack {
        LinkedList<Integer> stack = new LinkedList<>();

        // Push a new item into the stack
        public void push(int x) {
            stack.push(x);
        }

        // Pop the top of the stack
        public void pop() {
            stack.pop();
        }

        // Return the top of the stack
        public int top() {
            return stack.peekFirst();
        }

        // Check the stack is empty or not.
        public boolean isEmpty() {
            return stack.size() == 0;
        }
    }

    /**
     * Implement Stack by Two Queues.
     *
     * Implement a stack by two queues. The queue is first in first out (FIFO).
     * That means you can not directly pop the last element in a queue.
     */
    @tags.Stack
    @tags.Queue
    class StackTwoQueue {
        Queue<Integer> oldQ = new LinkedList<>();
        Queue<Integer> newQ = new LinkedList<>();

        // Push a new item into the stack
        public void push(int x) {
            newQ.offer(x);
            moveFromOldToNew();
        }

        // Pop the top of the stack
        public void pop() {
            oldQ.poll();
        }

        // Return the top of the stack
        public int top() {
            return oldQ.peek();
        }

        // Check the stack is empty or not.
        public boolean isEmpty() {
            return oldQ.isEmpty();
        }

        private void moveFromOldToNew() {
            while (!oldQ.isEmpty()) {
                newQ.offer(oldQ.poll());
            }
            Queue<Integer> temp = oldQ;
            oldQ = newQ;
            newQ = temp;
        }
    }

    /**
     * Implement Queue by Linked List.
     *
     * Implement a Queue by linked list. Support the following basic methods:
     * 1.enqueue(item). Put a new item in the queue. 2.dequeue(). Move the first
     * item out of the queue, return it.
     */
    @tags.Queue
    @tags.LinkedList
    public class QueueLinkedList {
        LinkedList<Integer> queue = new LinkedList<>();

        public QueueLinkedList() {
        }

        public void enqueue(int item) {
            queue.offer(item);
        }

        public int dequeue() {
            return queue.poll();
        }
    }

    /**
     * Implement Queue by Linked List II.
     *
     * Implement a Queue by linked list. Provide the following basic methods:
     * 1.push_front(item). Add a new item to the front of queue.
     * 2.push_back(item). Add a new item to the back of the queue.
     * 3.pop_front(). Move the first item out of the queue, return it.
     * 4.pop_back(). Move the last item out of the queue, return it.
     */
    @tags.LinkedList
    @tags.Queue
    public class Dequeue {
        LinkedList<Integer> list = new LinkedList<>();

        public Dequeue() {
        }

        public void push_front(int item) {
            list.offerFirst(item);
        }

        public void push_back(int item) {
            list.offer(item);
        }

        public int pop_front() {
            return list.poll();
        }

        public int pop_back() {
            return list.pollLast();
        }
    }

    // ---------------------------------------------------------------------- //
    // ---------------------------- Ugly Number ----------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Ugly Number.
     *
     * Write a program to check whether a given number is an ugly number.
     *
     * Ugly numbers are positive numbers whose prime factors only include 2, 3,
     * 5. For example, 6, 8 are ugly while 14 is not ugly since it includes
     * another prime factor 7.
     *
     * Notice: Note that 1 is typically treated as an ugly number.
     *
     * Example: Given num = 8 return true Given num = 14 return false.
     *
     * @param num
     *            an integer
     * @return true if num is an ugly number or false
     */
    @tags.Math
    @tags.Status.NeedPractice
    public boolean isUgly(int num) {
        if (num <= 0) {
            return false;
        }

        int[] primeFactors = { 2, 3, 5 };
        for (Integer prime : primeFactors) {
            while (num % prime == 0) {
                num /= prime;
            }
        }

        return num == 1;
    }

    /**
     * Ugly Number II - O(n) time.
     *
     * Ugly number is a number that only have factors 2, 3 and 5. Design an
     * algorithm to find the nth ugly number. The first 10 ugly numbers are 1,
     * 2, 3, 4, 5, 6, 8, 9, 10, 12...
     *
     * Notice: Note that 1 is typically treated as an ugly number.
     *
     * Example: If n=9, return 10.
     *
     * @param n
     *            an integer
     * @return the nth prime number as description.
     */
    @tags.Math
    @tags.Heap
    @tags.PriorityQueue
    @tags.DynamicProgramming
    @tags.Source.LintCode
    @tags.Status.NeedPractice
    public int nthUglyNumber(int n) {
        int[] uglyNums = new int[n];
        uglyNums[0] = 1;
        int ptr2 = 0, ptr3 = 0, ptr5 = 0;

        for (int i = 1; i < n; i++) {
            int n2 = uglyNums[ptr2] * 2;
            int n3 = uglyNums[ptr3] * 3;
            int n5 = uglyNums[ptr5] * 5;

            int min = Math.min(n2, n3);
            min = Math.min(min, n5);
            uglyNums[i] = min;

            if (min == n2) {
                ptr2++;
            }
            if (min == n3) {
                ptr3++;
            }
            if (min == n5) {
                ptr5++;
            }
        }

        return uglyNums[n - 1];
    }

    /**
     * Super Ugly Number.
     *
     * Write a program to find the nth super ugly number. Super ugly numbers are
     * positive numbers whose all prime factors are in the given prime list
     * primes of size k. For example, [1, 2, 4, 7, 8, 13, 14, 16, 19, 26, 28,
     * 32] is the sequence of the first 12 super ugly numbers given primes = [2,
     * 7, 13, 19] of size 4.
     *
     * Notice: 1 is a super ugly number for any given primes. The given numbers
     * in primes are in ascending order. 0 < k ¡Ü 100, 0 < n ¡Ü 10^6, 0 <
     * primes[i] < 1000 Have you met this question in a real interview? Yes
     *
     * Example: Given n = 6, primes = [2, 7, 13, 19] return 13
     *
     * @param n
     *            a positive integer
     * @param primes
     *            the given prime list
     * @return the nth super ugly number
     */
    @tags.Math
    @tags.Heap
    @tags.Company.Google
    @tags.Status.NeedPractice
    public int nthSuperUglyNumber(int n, int[] primes) {
        int k = primes.length;
        int[] ptrs = new int[k];
        int[] superUglyNums = new int[n];
        superUglyNums[0] = 1;

        for (int i = 1; i < n; i++) {
            int min = primes[0] * superUglyNums[ptrs[0]];
            for (int j = 1; j < k; j++) {
                min = Math.min(min, primes[j] * superUglyNums[ptrs[j]]);
            }
            superUglyNums[i] = min;

            for (int j = 0; j < k; j++) {
                if (min == primes[j] * superUglyNums[ptrs[j]]) {
                    ptrs[j]++;
                }
            }
        }

        return superUglyNums[n - 1];
    }

    // ---------------------------------------------------------------------- //
    // --------------------------- LRU/LFU Cache ---------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * LRU Cache.
     *
     * Design and implement a data structure for Least Recently Used (LRU)
     * cache. It should support the following operations: get and set.
     *
     * get(key) - Get the value (will always be positive) of the key if the key
     * exists in the cache, otherwise return -1.
     *
     * set(key, value) - Set or insert the value if the key is not already
     * present. When the cache reached its capacity, it should invalidate the
     * least recently used item before inserting a new item.
     */
    @tags.LinkedList
    @tags.Design
    @tags.Company.Amazon
    @tags.Company.Bloomberg
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Company.Microsoft
    @tags.Company.Palantir
    @tags.Company.Snapchat
    @tags.Company.Twitter
    @tags.Company.Uber
    @tags.Company.Yahoo
    @tags.Company.Zenefits
    @tags.Status.Hard
    public class LRUCache {
        private LinkedHashMap<Integer, Integer> map;
        private int cacheSize;
        private static final float hashTableLoadFactor = .75f; // default

        public LRUCache(int capacity) {
            this.cacheSize = capacity;
            map = new LinkedHashMap<Integer, Integer>(capacity,
                    hashTableLoadFactor, true) {
                private static final long serialVersionUID = 1L;

                protected boolean removeEldestEntry(
                        Map.Entry<Integer, Integer> eldest) {
                    return size() > LRUCache.this.cacheSize;
                }
            };
        }

        public int get(int key) {
            if (map.containsKey(key))
                return map.get(key);
            return -1;
        }

        public void set(int key, int value) {
            map.put(key, value);
        }
    }

    /** This is my solution which cannot promise O(1) for any operation */
    public class MyLRUCache {
        private HashMap<Integer, Integer> map;
        private int capacity;

        private LinkedList<Integer> queue;
        private int size;

        public MyLRUCache(int capacity) {
            map = new HashMap<Integer, Integer>(capacity);
            this.capacity = capacity;
            queue = new LinkedList<Integer>();
            size = 0;
        }

        public int get(int key) {
            if (!map.containsKey(key))
                return -1;
            moveKeyToLast(key);
            return map.get(key);
        }

        public void set(int key, int value) {
            if (map.containsKey(key)) {
                map.put(key, value);
                moveKeyToLast(key);
            } else {
                if (size < capacity) {
                    queue.add(key);
                    map.put(key, value);
                    size++;
                } else { // full
                    // remove old
                    int keyToDelete = queue.poll();
                    map.remove(keyToDelete);

                    // add new
                    queue.add(key);
                    map.put(key, value);
                }
            }
        }

        private void moveKeyToLast(int key) {
            for (int i = 0; i < queue.size(); i++) {
                if (queue.get(i) == key) {
                    queue.remove(i);
                    break;
                }
            }
            queue.add(key);
        }
    }

    /** Raw implementation of LinkedHashMap */
    @tags.LinkedList
    @tags.Design
    @tags.Company.Amazon
    @tags.Company.Bloomberg
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Company.Microsoft
    @tags.Company.Palantir
    @tags.Company.Snapchat
    @tags.Company.Twitter
    @tags.Company.Uber
    @tags.Company.Yahoo
    @tags.Company.Zenefits
    @tags.Status.NeedPractice
    public class MyLRUCacheRaw {
        class Node {
            int key;
            int value;
            Node prev;
            Node next;

            public Node(int key, int value) {
                this.key = key;
                this.value = value;
            }
        }

        private final Map<Integer, Node> map = new HashMap<>();
        private final int CAPACITY;
        private Node head;
        private Node tail;

        // @param capacity, an integer
        public MyLRUCacheRaw(int capacity) {
            CAPACITY = capacity;
        }

        // @return an integer
        public int get(int key) {
            Node node = map.get(key);
            if (node != null) {
                moveNodeToHead(node);
                return node.value;
            }
            return -1;
        }

        // @param key, an integer
        // @param value, an integer
        // @return nothing
        public void set(int key, int value) {
            Node node = map.get(key);
            if (node == null) {
                node = new Node(key, value);
                map.put(key, node);

                if (head == null) {
                    head = node;
                    tail = node;
                } else {
                    node.next = head;
                    head.prev = node;
                    head = node;
                }
            } else {
                node.value = value;
                moveNodeToHead(node);
            }

            if (map.size() > CAPACITY) {
                map.remove(tail.key);
                tail.prev.next = null;
                tail = tail.prev;
            }
        }

        private void moveNodeToHead(Node node) {
            if (node == head || head == tail) {
                return;
            }

            if (node == tail) {
                tail = tail.prev;
                tail.next = null;
            } else {
                node.prev.next = node.next;
                node.next.prev = node.prev;
            }
            node.next = head;
            head.prev = node;
            node.prev = null;
            head = node;
        }

    }

    /**
     * LFU Cache.
     *
     * LFU (Least Frequently Used) is a famous cache eviction algorithm. For a
     * cache with capacity k, if the cache is full and need to evict a key in
     * it, the key with the lease frequently used will be kicked out. Implement
     * set and get method for LFU cache.
     *
     * Example: Given capacity=3. set(2,2), set(1,1), get(2) >> 2, get(1) >> 1,
     * get(2) >> 2, set(3,3), set(4,4), get(3) >> -1, get(2) >> 2, get(1) >> 1,
     * get(4) >> 4.
     *
     * This is a naive solution, simply indicating a map is required for O(1)
     * lookup and priority queue is required for keeping a order of frequency.
     */
    @tags.PriorityQueue
    @tags.HashTable
    @tags.LinkedList
    @tags.Status.Hard
    public class LFUCache {
        class Node {
            int key;
            int value;
            int frequency;
            Node prev;
            Node next;
            long lastAccessTime;

            Node(int key, int value) {
                this.key = key;
                this.value = value;
                setTimeStamp();
            }

            int getValue() {
                frequency++;
                setTimeStamp();
                return value;
            }

            void setValue(int value) {
                frequency++;
                setTimeStamp();
                this.value = value;
            }

            private void setTimeStamp() {
                lastAccessTime = System.nanoTime();
            }
        }

        private final Map<Integer, Node> map = new HashMap<>();
        private final PriorityQueue<Node> queue;
        private final int CAPACITY;

        // @param capacity, an integer
        public LFUCache(int capacity) {
            CAPACITY = capacity;
            queue = new PriorityQueue<>(CAPACITY, new Comparator<Node>() {
                @Override
                public int compare(Node n1, Node n2) {
                    if (n1.frequency != n2.frequency) {
                        return n1.frequency - n2.frequency;
                    } else {
                        return Long.compare(n1.lastAccessTime,
                                n2.lastAccessTime);
                    }
                }
            });
        }

        // @param key, an integer
        // @param value, an integer
        // @return nothing
        public void set(int key, int value) {
            Node node = map.get(key);
            if (node != null) {
                node.setValue(value);
                queue.remove(node);
                queue.offer(node);
            } else {
                if (queue.size() == CAPACITY) {
                    node = queue.poll();
                    map.remove(node.key);
                }
                node = new Node(key, value);
                map.put(key, node);
                queue.offer(node);
            }
        }

        public int get(int key) {
            Node node = map.get(key);
            if (node != null) {
                int val = node.getValue();
                queue.remove(node);
                queue.offer(node);
                return val;
            }
            return -1;
        }
    }

    /*
     * Solution with all operations O(1) time (data structure only).
     */
    public class LFUCacheO1 {
        HashMap<Integer, Node> keyToNode;
        Node head;

        class Node {
            int frequency;
            HashMap<Integer, Integer> keyToValue;
            Node prev;
            Node next;
        }
    }

    // ---------------------------------------------------------------------- //
    // --------------------- Ascending/descending Stack --------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Min Stack.
     *
     * Implement a stack with min() function, which will return the smallest
     * number in the stack. It should support push, pop and min operation all in
     * O(1) cost.
     *
     * Notice: min operation will never be called if there is no number in the
     * stack.
     *
     * Example: push(1), pop() return 1, push(2), push(3), min() return 2,
     * push(1), min() return 1.
     */
    @tags.Stack
    @tags.Company.Google
    @tags.Company.Uber
    @tags.Company.Zenefits
    @tags.Status.NeedPractice
    public class MinStack {
        Stack<Integer> stack;
        Stack<Integer> minStack;

        public MinStack() {
            stack = new Stack<>();
            minStack = new Stack<>();
        }

        public void push(int number) {
            stack.push(number);
            if (minStack.isEmpty() || minStack.peek() >= number) { // empty case
                minStack.push(number);
            }
        }

        public int pop() {
            int number = stack.pop();
            if (minStack.peek() == number) { // easy to get this wrong
                minStack.pop();
            }
            return number;
        }

        public int min() {
            return minStack.peek();
        }
    }

    /**
     * Largest Rectangle in Histogram.
     *
     * Given n non-negative integers representing the histogram's bar height
     * where the width of each bar is 1, find the area of largest rectangle in
     * the histogram.
     *
     * Above is a histogram where width of each bar is 1, given height =
     * [2,1,5,6,2,3]. The largest rectangle is shown in the shaded area, which
     * has area = 10 unit.
     *
     * Example: Given height = [2,1,5,6,2,3], return 10.
     *
     * What a elegant solution!
     *
     * @param height:
     *            A list of integer
     * @return: The area of largest rectangle in the histogram
     */
    @tags.Stack
    @tags.Array
    @tags.Status.Hard
    public int largestRectangleArea(int[] height) {
        if (height == null || height.length == 0) {
            return 0;
        }

        Stack<Integer> stack = new Stack<Integer>();
        int max = 0;

        for (int i = 0; i <= height.length; i++) {
            int currentH = (i == height.length) ? -1 : height[i];
            while (!stack.isEmpty() && currentH < height[stack.peek()]) {
                int h = height[stack.pop()];
                int start = stack.isEmpty() ? -1 : stack.peek();
                max = Math.max(max, (i - start - 1) * h);
            }

            stack.push(i);
        }
        return max;
    }

    /**
     * Maximal Rectangle.
     *
     * Given a 2D boolean matrix filled with False and True, find the largest
     * rectangle containing all True and return its area.
     *
     * Example: Given a matrix:
     * [[1,1,0,0,1],[0,1,0,0,1],[0,0,1,1,1],[0,0,1,1,1],[0,0,0,0,1]] return 6.
     */
    @tags.Stack
    @tags.Array
    @tags.DynamicProgramming
    public int maximalRectangle(boolean[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }

        int m = matrix.length, n = matrix[0].length;
        int[] histogram = new int[n];
        int max = 0;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                histogram[j] = matrix[i][j] ? histogram[j] + 1 : 0;
            }
            max = Math.max(max, largestRectangleArea(histogram));
        }

        return max;
    }

    /**
     * Maximal Square.
     *
     * Given a 2D binary matrix filled with 0's and 1's, find the largest square
     * containing all 1's and return its area.
     *
     * Example For example, given the following matrix:
     * [[1,0,1,0,0],[1,0,1,1,1],[1,1,1,1,1],[1,0,0,1,0]] Return 4.
     *
     * @param matrix:
     *            a matrix of 0 and 1
     * @return: an integer
     */
    @tags.Stack
    @tags.Array
    @tags.DynamicProgramming
    @tags.Company.Airbnb
    @tags.Company.Facebook
    public int maxSquare(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }

        int m = matrix.length, n = matrix[0].length;
        int[] histogram = new int[n];
        int max = 0;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                histogram[j] = matrix[i][j] == 0 ? 0 : histogram[j] + 1;
            }
            max = Math.max(max, largestSquareArea(histogram));
        }

        return max;
    }

    private int largestSquareArea(int[] histogram) {
        Stack<Integer> stack = new Stack<>();
        int max = 0;

        for (int i = 0; i <= histogram.length; i++) {
            int currentH = (i == histogram.length) ? -1 : histogram[i];
            while (!stack.isEmpty() && currentH < histogram[stack.peek()]) {
                int h = histogram[stack.pop()];
                int start = stack.isEmpty() ? -1 : stack.peek();
                int width = i - start - 1;
                int side = Math.min(h, width);

                max = Math.max(max, side * side);
            }

            stack.push(i);
        }

        return max;
    }

    /**
     * Max Tree.
     *
     * Given an integer array with no duplicates. A max tree building on this
     * array is defined as follow: The root is the maximum number in the array;
     * The left subtree and right subtree are the max trees of the subarray
     * divided by the root number. Construct the max tree by the given array.
     *
     * Example: Given [2, 5, 6, 0, 3, 1], the max tree constructed by this array
     * is:
     *
     *     6
     *    / \
     *   5   3
     *  /   / \
     * 2   0   1
     *
     * Challenge: O(n) time and memory.
     *
     * @param A:
     *            Given an integer array with no duplicates.
     * @return: The root of max tree.
     */
    @tags.Stack
    @tags.CartesianTree
    @tags.Source.LintCode
    public TreeNode maxTree(int[] A) {
        if (A == null || A.length == 0) {
            return null;
        }

        Stack<TreeNode> stack = new Stack<>();
        for (int i = 0; i <= A.length; i++) {
            int currentVal = (i == A.length) ? Integer.MAX_VALUE : A[i];
            TreeNode current = new TreeNode(currentVal);

            while (!stack.isEmpty() && current.val > stack.peek().val) {
                TreeNode left = stack.pop();
                if (!stack.isEmpty() && current.val > stack.peek().val) {
                    stack.peek().right = left;
                } else {
                    current.left = left;
                }
            }
            stack.push(current);
        }

        return stack.peek().left;
    }

    // ---------------------------------------------------------------------- //
    // -------------------------------- Trie -------------------------------- //
    // ---------------------------------------------------------------------- //
    
    /**
     * Word Search.
     *
     * Given a 2D board and a word, find if the word exists in the grid.
     *
     * The word can be constructed from letters of sequentially adjacent cell,
     * where "adjacent" cells are those horizontally or vertically neighboring.
     * The same letter cell may not be used more than once.
     *
     * Example Given board = [ "ABCE", "SFCS", "ADEE" ]
     *
     * word = "ABCCED", -> returns true,
     * word = "SEE", -> returns true,
     * word = "ABCB", -> returns false.
     *
     * @param board: A list of lists of character
     * @param word: A string
     * @return: A boolean
     */
    @tags.Array
    @tags.DFS
    @tags.Backtracking
    @tags.Company.Bloomberg
    @tags.Company.Facebook
    @tags.Company.Microsoft
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0 || board[0].length == 0) {
            return false;
        }

        int m = board.length, n = board[0].length;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (exist(board, i, j, word, 0)) {
                    return true;
                }
            }
        }

        return false;
    }

    private boolean exist(char[][] board, int i, int j, String word, int pos) {
        if (pos == word.length()) {
            return true;
        }

        // index out of board
        int m = board.length, n = board[0].length;
        if (i < 0 || i >= m || j < 0 || j >= n) {
            return false;
        }

        // mismatch
        char c = board[i][j];
        if (c != word.charAt(pos)) {
            return false;
        }

        // down, right, up, left
        int[] xs = { 1, 0, -1, 0 };
        int[] ys = { 0, 1, 0, -1 };

        for (int k = 0; k < 4; k++) {
            board[i][j] = '#';
            if (exist(board, i + xs[k], j + ys[k], word, pos + 1)) {
                return true;
            }
            board[i][j] = c;
        }

        return false;
    }

    /**
     * Word Search II.
     *
     * Given a matrix of lower alphabets and a dictionary. Find all words in the
     * dictionary that can be found in the matrix. A word can start from any
     * position in the matrix and go left/right/up/down to the adjacent
     * position.
     *
     * Example: Given matrix: ["doaf","agai","dcan"], and dictionary: {"dog",
     * "dad", "dgdg", "can", "again"}, return {"dog", "dad", "can", "again"}.
     *
     * Challenge: Using trie to implement your algorithm.
     *
     * @param board:
     *            A list of lists of character
     * @param words:
     *            A list of string
     * @return: A list of string
     */
    @tags.Trie
    @tags.Backtracking
    @tags.Source.LintCode
    @tags.Company.Airbnb
    @tags.Company.Google
    @tags.Company.Microsoft
    @tags.Status.Hard
    public ArrayList<String> wordSearchII(char[][] board,
            ArrayList<String> words) {
        ArrayList<String> result = new ArrayList<>();
        if (board == null || board.length == 0 || board[0].length == 0) {
            return result;
        }

        // load dictionary in Trie tree
        Trie root = new Trie();
        for (String word : words) {
            addToTrie(word, root);
        }

        // search the board
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                searchWord(board, i, j, root, result, new StringBuilder());
            }
        }

        return result;
    }

    private void addToTrie(String word, Trie root) {
        for (int i = 0; i < word.length(); i++) {
            int index = word.charAt(i) - 'a';
            if (root.letters[index] == null) {
                root.letters[index] = new Trie();
            }
            root = root.letters[index];
        }
        root.isWord = true;
    }

    private void searchWord(char[][] board, int i, int j, Trie root,
            ArrayList<String> result, StringBuilder sb) {
        if (i < 0 || i == board.length || j < 0 || j == board[0].length) {
            return;
        }

        char c = board[i][j];
        int pos = c - 'a';
        if (pos < 0 || pos >= 26 || root.letters[pos] == null) {
            return;
        }
        root = root.letters[pos];

        sb.append(c);
        board[i][j] = '0';
        if (root.isWord && !result.contains(sb.toString())) {
            result.add(sb.toString());
        }

        searchWord(board, i - 1, j, root, result, sb);
        searchWord(board, i + 1, j, root, result, sb);
        searchWord(board, i, j - 1, root, result, sb);
        searchWord(board, i, j + 1, root, result, sb);

        sb.deleteCharAt(sb.length() - 1);
        board[i][j] = c;
    }

    class Trie {
        Trie[] letters = new Trie[26];
        boolean isWord;
    }

    // ---------------------------------------------------------------------- //
    // -------------------------- Range Sum Query --------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Range Sum Query - Immutable.
     *
     * Given an integer array nums, find the sum of the elements between indices
     * i and j (i ¡Ü j), inclusive.
     *
     * Example: Given nums = [-2, 0, 3, -5, 2, -1].
     *
     * sumRange(0, 2) -> 1. sumRange(2, 5) -> -1. sumRange(0, 5) -> -3.
     *
     * Note: You may assume that the array does not change. There are many calls
     * to sumRange function.
     */
    @tags.DynamicProgramming
    @tags.Company.Palantir
    public class NumArray {
        // Your NumArray object will be instantiated and called as such:
        // NumArray numArray = new NumArray(nums);
        // numArray.sumRange(0, 1);
        // numArray.sumRange(1, 2);

        int[] sum; // sum to the end

        public NumArray(int[] nums) {
            if (nums == null || nums.length == 0) {
                sum = new int[1];
            }
            sum = new int[nums.length + 1];
            for (int i = nums.length - 1; i >= 0; i--) {
                sum[i] = nums[i] + sum[i + 1];
            }
        }

        public int sumRange(int i, int j) {
            return sum[i] - sum[j + 1];
        }
    }

    /**
     * Range Sum Query - Mutable.
     *
     * Given an integer array nums, find the sum of the elements between indices
     * i and j (i ¡Ü j), inclusive.
     *
     * The update(i, val) function modifies nums by updating the element at
     * index i to val. Example: Given nums = [1, 3, 5]
     *
     * sumRange(0, 2) -> 9 update(1, 2) sumRange(0, 2) -> 8 Note: The array is
     * only modifiable by the update function. You may assume the number of
     * calls to update and sumRange function is distributed evenly.
     */
    @tags.SegmentTree
    @tags.BinaryIndexedTree
    public class NumArrayII {
        // Your NumMatrix object will be instantiated and called as such:
        // NumMatrix numMatrix = new NumMatrix(matrix);
        // numMatrix.sumRegion(0, 1, 2, 3);
        // numMatrix.sumRegion(1, 2, 3, 4);

        public NumArrayII(int[] nums) {
            // TODO
        }

        void update(int i, int val) {

        }

        public int sumRange(int i, int j) {
            return 0;
        }
    }

    /**
     * Range Sum Query 2D - Immutable.
     *
     * Given a 2D matrix matrix, find the sum of the elements inside the
     * rectangle defined by its upper left corner (row1, col1) and lower right
     * corner (row2, col2).
     *
     * Example: Given matrix = [ [3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1,
     * 5], [4, 1, 0, 1, 7], [1, 0, 3, 0, 5] ].
     *
     * sumRegion(2, 1, 4, 3) -> 8. sumRegion(1, 1, 2, 2) -> 11. sumRegion(1, 2,
     * 2, 4) -> 12.
     *
     * Note: You may assume that the matrix does not change. There are many
     * calls to sumRegion function. You may assume that row1 ¡Ü row2 and col1 ¡Ü
     * col2.
     */
    @tags.DynamicProgramming
    @tags.Company.Cloudera
    public class NumMatrix {
        // Your NumMatrix object will be instantiated and called as such:
        // NumMatrix numMatrix = new NumMatrix(matrix);
        // numMatrix.sumRegion(0, 1, 2, 3);
        // numMatrix.sumRegion(1, 2, 3, 4);

        int[][] sum; // sum to the right bottom corner

        public NumMatrix(int[][] matrix) {
            if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
                sum = new int[1][1];
                return;
            }

            int m = matrix.length, n = matrix[0].length;
            sum = new int[m + 1][n + 1];
            sum[m - 1][n - 1] = matrix[m - 1][n - 1];

            for (int i = m - 1; i >= 0; i--) {
                for (int j = n - 1; j >= 0; j--) {
                    sum[i][j] = matrix[i][j] + sum[i + 1][j] + sum[i][j + 1]
                            - sum[i + 1][j + 1];
                }
            }
        }

        public int sumRegion(int row1, int col1, int row2, int col2) {
            return sum[row1][col1] + sum[row2 + 1][col2 + 1]
                    - sum[row1][col2 + 1] - sum[row2 + 1][col1];
        }
    }

    /**
     * Range Sum Query 2D - Mutable.
     *
     * Given a 2D matrix matrix, find the sum of the elements inside the
     * rectangle defined by its upper left corner (row1, col1) and lower right
     * corner (row2, col2).
     *
     * Example: Given matrix = [ [3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1,
     * 5], [4, 1, 0, 1, 7], [1, 0, 3, 0, 5] ]
     *
     * sumRegion(2, 1, 4, 3) -> 8. update(3, 2, 2). sumRegion(2, 1, 4, 3) -> 10.
     * Note: The matrix is only modifiable by the update function. You may
     * assume the number of calls to update and sumRegion function is
     * distributed evenly. You may assume that row1 ¡Ü row2 and col1 ¡Ü col2.
     */
    @tags.SegmentTree
    @tags.BinaryIndexedTree
    @tags.Company.Google
    public class NumMatrixII {
        // Your NumMatrix object will be instantiated and called as such:
        // NumMatrix numMatrix = new NumMatrix(matrix);
        // numMatrix.sumRegion(0, 1, 2, 3);
        // numMatrix.update(1, 1, 10);
        // numMatrix.sumRegion(1, 2, 3, 4);

        public NumMatrixII(int[][] matrix) {
            // TODO
        }

        public void update(int row, int col, int val) {

        }

        public int sumRegion(int row1, int col1, int row2, int col2) {
            return 0;
        }
    }

    /**
     * The Skyline Problem (Building Outline).
     *
     * A city's skyline is the outer contour of the silhouette formed by all the
     * buildings in that city when viewed from a distance. Now suppose you are
     * given the locations and height of all the buildings as shown on a
     * cityscape photo (Figure A), write a program to output the skyline formed
     * by these buildings collectively.
     *
     * Buildings Skyline Contour The geometric information of each building is
     * represented by a triplet of integers [Li, Ri, Hi], where Li and Ri are
     * the x coordinates of the left and right edge of the ith building,
     * respectively, and Hi is its height. It is guaranteed that 0 ¡Ü Li, Ri ¡Ü
     * INT_MAX, 0 < Hi ¡Ü INT_MAX, and Ri - Li > 0. You may assume all buildings
     * are perfect rectangles grounded on an absolutely flat surface at height
     * 0.
     *
     * For instance, the dimensions of all buildings in Figure A are recorded
     * as: [ [2 9 10], [3 7 15], [5 12 12], [15 20 10], [19 24 8] ] .
     *
     * The output is a list of "key points" (red dots in Figure B) in the format
     * of [ [x1,y1], [x2, y2], [x3, y3], ... ] that uniquely defines a skyline.
     * A key point is the left endpoint of a horizontal line segment. Note that
     * the last key point, where the rightmost building ends, is merely used to
     * mark the termination of the skyline, and always has zero height. Also,
     * the ground in between any two adjacent buildings should be considered
     * part of the skyline contour.
     *
     * For instance, the skyline in Figure B should be represented as:[ [2 10],
     * [3 15], [7 12], [12 0], [15 10], [20 8], [24, 0] ].
     *
     * Notes: The number of buildings in any input list is guaranteed to be in
     * the range [0, 10000]. The input list is already sorted in ascending order
     * by the left x position Li. The output list must be sorted by the x
     * position. There must be no consecutive horizontal lines of equal height
     * in the output skyline. For instance, [...[2 3], [4 5], [7 5], [11 5], [12
     * 7]...] is not acceptable; the three lines of height 5 should be merged
     * into one in the final output as such: [...[2 3], [4 5], [12 7], ...]
     *
     * @param buildings: A list of lists of integers
     * @return: Find the outline of those buildings
     */
    @tags.Heap
    @tags.BinaryIndexedTree
    @tags.SegmentTree
    @tags.DivideAndConquer
    @tags.Source.LeetCode
    @tags.Source.LintCode
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Company.Microsoft
    @tags.Company.Twitter
    @tags.Company.Yelp
    public List<int[]> getSkyline(int[][] buildings) {
        // TODO this is not a binary indexed tree solution

        List<int[]> skyline = new ArrayList<>();
        if (buildings == null || buildings.length == 0) {
            return skyline;
        }

        class Point {
            int x, h;
            boolean start;

            public Point(int x, int h, boolean start) {
                this.x = x;
                this.h = h;
                this.start = start;
            }
        }

        List<Point> list = new ArrayList<>();
        for (int[] building : buildings) {
            list.add(new Point(building[0], building[2], true));
            list.add(new Point(building[1], building[2], false));
        }

        Collections.sort(list, new Comparator<Point>() {
            @Override
            public int compare(Point p1, Point p2) {
                if (p1.x != p2.x) {
                    return p1.x - p2.x;
                } else {
                    if (p2.start && !p1.start) {
                        return 1;
                    }
                    return -1;
                }
            }
        });

        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(buildings.length,
                Collections.reverseOrder());
        List<int[]> tmp = new ArrayList<>();
        for (Point p : list) {
            boolean ending = !maxHeap.isEmpty();
            if (p.start) {
                maxHeap.offer(p.h);
            } else {
                maxHeap.remove(p.h);
            }

            if (!maxHeap.isEmpty()) {
                tmp.add(new int[] { p.x, maxHeap.peek() });
            } else if (ending) {
                tmp.add(new int[] { p.x, 0 });
            }
        }

        skyline.add(tmp.get(0));
        for (int i = 1; i < tmp.size(); i++) {
            int[] figure = tmp.get(i);
            int[] prev = skyline.get(skyline.size() - 1);
            if (prev[0] == figure[0]) {
                if (figure[1] == 0) {
                    prev[1] = 0;
                } else {
                    prev[1] = Math.max(prev[1], figure[1]);
                }
            } else {
                if (figure[1] != prev[1]) {
                    skyline.add(figure);
                }
            }
        }

        return skyline;
    }

    // ---------------------------------------------------------------------- //
    // ------------------------------ Unit Tests ---------------------------- //
    // ---------------------------------------------------------------------- //

    @Test
    public void test() {
        hashCodeTest();
        isUglyTest();
        nthSuperUglyNumberTest();
        LFUCacheTest();
        largestRectangleAreaTest();
    }

    private void hashCodeTest() {
        char[] key = "abcdefghijklmnopqrstuvwxyz".toCharArray();
        int hashSize = 2607;
        Assert.assertEquals(1673, hashCode(key, hashSize));

        key = "Wrong answer or accepted?".toCharArray();
        hashSize = 1000000007;
        Assert.assertEquals(382528955, hashCode(key, hashSize));
    }

    private void isUglyTest() {
        Assert.assertTrue(isUgly(8));
        Assert.assertFalse(isUgly(14));
    }

    private void nthSuperUglyNumberTest() {
        int n = 45;
        int[] primes = {2,3,7,13,17,23,31,41,43,47};
        Assert.assertEquals(82, nthSuperUglyNumber(n, primes));
    }

    private void LFUCacheTest() {
        int capacity = 3;
        LFUCache lfu = new LFUCache(capacity);

        lfu.set(1, 10);
        lfu.set(2, 20);
        lfu.set(3, 30);
        Assert.assertEquals(10, lfu.get(1));
        lfu.set(4, 40);
        Assert.assertEquals(40, lfu.get(4));
        Assert.assertEquals(30, lfu.get(3));
        Assert.assertEquals(-1, lfu.get(2));
        Assert.assertEquals(10, lfu.get(1));
        lfu.set(5, 50);
        Assert.assertEquals(10, lfu.get(1));
        Assert.assertEquals(-1, lfu.get(2));
        Assert.assertEquals(30, lfu.get(3));
        Assert.assertEquals(-1, lfu.get(4));
        Assert.assertEquals(50, lfu.get(5));
    }

    private void largestRectangleAreaTest() {
        int[] histogram1 = { 1, 1 };
        Assert.assertEquals(2, largestRectangleArea(histogram1));

        int[] histogram2 = { 5, 4, 1, 2 };
        Assert.assertEquals(8, largestRectangleArea(histogram2));

        int[] histogram3 = { 2, 1, 5, 6, 2, 3 };
        Assert.assertEquals(10, largestRectangleArea(histogram3));
    }
}
