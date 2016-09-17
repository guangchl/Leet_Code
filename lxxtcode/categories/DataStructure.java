package categories;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
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
    @tags.Status.SuperHard
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
        PriorityQueue<ListNode> pq = new PriorityQueue<ListNode>(lists.size(),
                new Comparator<ListNode>() {
                    @Override
                    public int compare(ListNode n1, ListNode n2) {
                        return n1.val - n2.val;
                    }
                });

        // insert head node of each list
        for (ListNode list : lists) {
            if (list != null) {
                pq.offer(list);
            }
        }

        // merge
        ListNode dummy = new ListNode(0);
        ListNode prev = dummy;
        while (!pq.isEmpty()) {
            prev.next = pq.poll();
            prev = prev.next;
            if (prev.next != null) {
                pq.offer(prev.next);
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
            return null;
        }

        int n = nums.length;
        PriorityQueue<Integer> minHeap = new PriorityQueue<>(n / 2 + 1);
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(n / 2 + 1,
                new Comparator<Integer>() {
                    @Override
                    public int compare(Integer i1, Integer i2) {
                        return i2 - i1;
                    }
                });

        int[] medians = new int[n];
        for (int i = 0; i < nums.length; i++) {
            if (minHeap.size() == maxHeap.size()) {
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
        PriorityQueue<Integer> more = new PriorityQueue<>();
        PriorityQueue<Integer> less = new PriorityQueue<>(n / 2 + 1,
                new Comparator<Integer>() {
                    @Override
                    public int compare(Integer i1, Integer i2) {
                        return i2 - i1;
                    }
                });

        // init 2 min heap with balanced count
        for (int i = 0; i < k; i++) {
            more.offer(nums[i]);
        }
        for (int i = 0; i < (k + 1) / 2; i++) {
            less.offer(more.poll());
        }

        // sliding
        medians.add(less.peek());
        for (int i = k; i < n; i++) {
            if (less.remove(nums[i - k])) {
                more.offer(nums[i]);
                less.offer(more.poll());
            } else {
                more.remove(nums[i - k]);
                less.offer(nums[i]);
                more.offer(less.poll());
            }
            medians.add(less.peek());
        }

        return medians;
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
        List<Integer> uglyNums = new ArrayList<>();
        uglyNums.add(1);
        int ptr2 = 0, ptr3 = 0, ptr5 = 0;

        while (uglyNums.size() < n) {
            int n2 = uglyNums.get(ptr2) * 2;
            int n3 = uglyNums.get(ptr3) * 3;
            int n5 = uglyNums.get(ptr5) * 5;
            int min = Math.min(n2, n3);
            min = Math.min(min, n5);
            uglyNums.add(min);
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

        return uglyNums.get(n - 1);
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
     * LRU Cache
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
    @tags.Company.Google
    @tags.Company.Uber
    @tags.Company.Zenefits
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
            if (minStack.isEmpty() || minStack.peek() >= number) {
                minStack.push(number);
            }
        }

        public int pop() {
            int number = stack.pop();
            if (minStack.peek() == number) {
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
    @tags.Status.SuperHard
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
    @tags.Source.LintCode
    @tags.Company.Airbnb
    @tags.Status.SuperHard
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
    // ------------------------------ Unit Tests ---------------------------- //
    // ---------------------------------------------------------------------- //

    @Test
    public void test() {
        hashCodeTest();
        isUglyTest();
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
