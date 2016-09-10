package categories;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;

import org.junit.Test;
import org.junit.Assert;

public class GraphAndSearch {

    // ---------------------------------------------------------------------- //
    // ------------------------------- MODELS ------------------------------- //
    // ---------------------------------------------------------------------- //

    /** Definition for undirected graph. */
    class UndirectedGraphNode {
        int label;
        ArrayList<UndirectedGraphNode> neighbors;

        UndirectedGraphNode(int x) {
            label = x;
            neighbors = new ArrayList<UndirectedGraphNode>();
        }
    }

    /** Definition for Directed graph. */
    class DirectedGraphNode {
        int label;
        ArrayList<DirectedGraphNode> neighbors;

        DirectedGraphNode(int x) {
            label = x;
            neighbors = new ArrayList<DirectedGraphNode>();
        }
    }

    // ---------------------------------------------------------------------- //
    // ------------------------------ PROBLEMS ------------------------------ //
    // ---------------------------------------------------------------------- //

    /**
     * Six Degrees.
     *
     * Six degrees of separation is the theory that everyone and everything is
     * six or fewer steps away, by way of introduction, from any other person in
     * the world, so that a chain of "a friend of a friend" statements can be
     * made to connect any two people in a maximum of six steps. Given a
     * friendship relations, find the degrees of two people, return -1 if they
     * can not been connected by friends of friends.
     *
     * @param graph
     *            a list of Undirected graph node
     * @param s,
     *            t two Undirected graph nodes
     * @return an integer
     */
    @tags.Company.Microsoft
    @tags.BFS
    public int sixDegrees(List<UndirectedGraphNode> graph,
            UndirectedGraphNode s, UndirectedGraphNode t) {
        Set<UndirectedGraphNode> visited = new HashSet<>();
        Queue<UndirectedGraphNode> queue = new LinkedList<>();
        queue.offer(s);
        int degree = 0;

        while (!queue.isEmpty()) {
            Queue<UndirectedGraphNode> next = new LinkedList<>();
            while (!queue.isEmpty()) {
                UndirectedGraphNode node = queue.poll();

                if (node != null && !visited.contains(node)) {
                    if (node == t) {
                        return degree;
                    }

                    visited.add(node);

                    for (UndirectedGraphNode neighbor : node.neighbors) {
                        next.offer(neighbor);
                    }
                }
            }
            queue = next;
            degree++;
        }

        return -1;
    }

    /**
     * Route Between Two Nodes in Graph.
     *
     * Given a directed graph, design an algorithm to find out whether there is
     * a route between two nodes.
     *
     * @param graph:
     *            A list of Directed graph node
     * @param s:
     *            the starting Directed graph node
     * @param t:
     *            the terminal Directed graph node
     * @return: a boolean value
     */
    @tags.DFS
    @tags.BFS
    @tags.Source.CrackingTheCodingInterview
    public boolean hasRoute(ArrayList<DirectedGraphNode> graph,
            DirectedGraphNode s, DirectedGraphNode t) {
        Set<DirectedGraphNode> set = new HashSet<>();
        Stack<DirectedGraphNode> stack = new Stack<>();
        stack.push(s);

        while (!stack.isEmpty()) {
            DirectedGraphNode node = stack.pop();
            if (node != null && !set.contains(node)) {
                if (node == t) {
                    return true;
                }
                for (DirectedGraphNode neighbor : node.neighbors) {
                    stack.push(neighbor);
                }
            }
        }

        return false;
    }

    /**
     * Clone Graph.
     *
     * Clone an undirected graph. Each node in the graph contains a label and a
     * list of its neighbors. Return a deep copied graph.
     *
     * @param node:
     *            A undirected graph node
     * @return: A undirected graph node
     */
    @tags.Graph
    @tags.DFS
    @tags.BFS
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Company.PocketGems
    @tags.Company.Uber
    @tags.Status.NeedPractice
    public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
        Map<UndirectedGraphNode, UndirectedGraphNode> map = new HashMap<>();
        Set<UndirectedGraphNode> visited = new HashSet<>();
        Stack<UndirectedGraphNode> stack = new Stack<>();
        stack.push(node);

        while (!stack.isEmpty()) {
            UndirectedGraphNode n = stack.pop();

            if (n != null && !visited.contains(n)) {
                visited.add(n);

                // copy node
                UndirectedGraphNode copy = map.get(n);
                if (copy == null) {
                    copy = new UndirectedGraphNode(n.label);
                    map.put(n, copy);
                }

                // copy neighbors
                for (UndirectedGraphNode neighbor : n.neighbors) {
                    stack.push(neighbor);
                    if (map.containsKey(neighbor)) {
                        copy.neighbors.add(map.get(neighbor));
                    } else {
                        UndirectedGraphNode neighborCopy = new UndirectedGraphNode(
                                neighbor.label);
                        map.put(neighbor, neighborCopy);
                        copy.neighbors.add(neighborCopy);
                    }
                }
            }
        }

        return map.get(node);
    }

    /**
     * Clone Graph.
     *
     * This is a nice recursive solution. Another iterative solution: copy node
     * first, then copy connection use map.
     */
    @tags.DFS
    @tags.BFS
    @tags.Company.Facebook
    public UndirectedGraphNode cloneGraphRecursive(UndirectedGraphNode node) {
        HashMap<UndirectedGraphNode, UndirectedGraphNode> map = new HashMap<UndirectedGraphNode, UndirectedGraphNode>();
        return cloneNode(node, map);
    }

    private UndirectedGraphNode cloneNode(UndirectedGraphNode node,
            HashMap<UndirectedGraphNode, UndirectedGraphNode> map) {
        if (node == null)
            return null;
        if (map.containsKey(node)) { // have copied before
            return map.get(node);
        } else { // hasn't been copied
            UndirectedGraphNode copy = new UndirectedGraphNode(node.label);
            map.put(node, copy); // put the new copy into map
            // add copies of children
            for (UndirectedGraphNode n : node.neighbors) {
                copy.neighbors.add(cloneNode(n, map));
            }
            return copy;
        }
    }

    /**
     * Topological Sorting.
     *
     * Given an directed graph, a topological order of the graph nodes is
     * defined as follow: For each directed edge A -> B in graph, A must before
     * B in the order list. The first node in the order can be any node in the
     * graph with no nodes direct to it. Find any topological order for the
     * given graph.
     *
     * Notice: You can assume that there is at least one topological order in
     * the graph.
     *
     * Challenge: Can you do it in both BFS and DFS?
     *
     * @param graph:
     *            A list of Directed graph node
     * @return: Any topological order for the given graph.
     */
    @tags.BFS
    @tags.DFS
    @tags.Source.GeeksForGeeks
    @tags.Source.LintCode
    @tags.Status.SuperHard
    public ArrayList<DirectedGraphNode> topSort(
            ArrayList<DirectedGraphNode> graph) {
        Map<DirectedGraphNode, Integer> refCount = new HashMap<>();
        for (DirectedGraphNode node : graph) {
            for (DirectedGraphNode neighbor : node.neighbors) {
                if (refCount.containsKey(neighbor)) {
                    refCount.put(neighbor, refCount.get(neighbor) + 1);
                } else {
                    refCount.put(neighbor, 1);
                }
            }
        }

        ArrayList<DirectedGraphNode> result = new ArrayList<>();
        Queue<DirectedGraphNode> queue = new LinkedList<>();
        for (DirectedGraphNode node : graph) {
            if (!refCount.containsKey(node)) {
                queue.offer(node);
                result.add(node);
            }
        }

        while (!queue.isEmpty()) {
            // add neighbor with ref count 1
            DirectedGraphNode node = queue.poll();
            for (DirectedGraphNode neighbor : node.neighbors) {
                if (refCount.get(neighbor) == 1) {
                    queue.offer(neighbor);
                    result.add(neighbor);
                }
                refCount.put(neighbor, refCount.get(neighbor) - 1);
            }
        }

        return result;
    }

    // ---------------------------------------------------------------------- //
    // ---------------------------- Combinations ---------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Combinations.
     *
     * Given two integers n and k, return all possible combinations of k numbers
     * out of 1 ... n.
     *
     * Example: For example, If n = 4 and k = 2, a solution is:
     * [[2,4],[3,4],[2,3],[1,2],[1,3],[1,4]].
     *
     * @param n: Given the range of numbers
     * @param k: Given the numbers of combinations
     * @return: All the combinations of k numbers out of 1..n
     */
    @tags.Array
    @tags.Backtracking
    @tags.DFS
    public ArrayList<ArrayList<Integer>> combine(int n, int k) {
        ArrayList<ArrayList<Integer>> combinations = new ArrayList<ArrayList<Integer>>();

        if (n == 0 || k == 0 || n < k) {
            return combinations;
        }

        combinations.add(new ArrayList<Integer>());

        for (int i = 1; i <= n; i++) {
            int len = combinations.size();
            // add new lists that contain i for lists that are not full
            for (int j = 0; j < len; j++) {
                ArrayList<Integer> oldList = combinations.get(j);

                // list that not full
                if (oldList.size() < k) {
                    // list that must contain all last integers
                    if (k - oldList.size() == n - i + 1) {
                        // add all last integers to the list
                        for (int num = i; num <= n; num++) {
                            oldList.add(num);
                        }
                    } else {
                        // copy the old list and add i to it,
                        // then add the new list to the combinations
                        ArrayList<Integer> newList = new ArrayList<Integer>(
                                oldList);
                        newList.add(i);
                        combinations.add(newList);
                    }
                }
            }
        }

        return combinations;
    }

    /** Combinations - recursive solution. */
    @tags.Array
    @tags.Backtracking
    @tags.DFS
    public List<List<Integer>> combine2(int n, int k) {
        List<List<Integer>> result = new ArrayList<>();
        combine(n, 1, k, result, new ArrayList<Integer>());
        return result;
    }

    private void combine(int n, int num, int k, List<List<Integer>> result,
            List<Integer> path) {
        if (k == 0) {
            result.add(new ArrayList<>(path));
            return;
        } else if (num > n) {
            return;
        }

        path.add(num);
        combine(n, num + 1, k - 1, result, path);
        path.remove(path.size() - 1);
        combine(n, num + 1, k, result, path);
    }

    /**
     * Combination Sum.
     *
     * Given a set of candidate numbers (C) and a target number (T), find all
     * unique combinations in C where the candidate numbers sums to T. The same
     * repeated number may be chosen from C unlimited number of times.
     *
     * Notice: All numbers (including target) will be positive integers.
     * Elements in a combination (a1, a2, ¡­ , ak) must be in non-descending
     * order. (ie, a1 ¡Ü a2 ¡Ü ¡­ ¡Ü ak). The solution set must not contain
     * duplicate combinations.
     *
     * For example, given candidate set 2,3,6,7 and target 7, A solution set is:
     * [7] [2, 2, 3]
     *
     * @param candidates:
     *            A list of integers
     * @param target:An
     *            integer
     * @return: A list of lists of integers
     */
    @tags.DFS
    @tags.Backtracking
    @tags.Array
    @tags.Company.Snapchat
    @tags.Company.Uber
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        if (candidates == null || candidates.length == 0) {
            return Collections.emptyList();
        }

        List<List<Integer>> result = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        Arrays.sort(candidates);
        combinationSum(candidates, target, 0, path, result);
        return result;
    }

    private void combinationSum(int[] candidates, int target, int pos,
            List<Integer> path, List<List<Integer>> result) {
        if (target == 0) {
            result.add(new ArrayList<>(path));
        } else if (target < 0) {
            return;
        }
        for (int i = pos; i < candidates.length; i++) {
            if (i > 0 && candidates[i] == candidates[i - 1]) {
                continue;
            }
            path.add(candidates[i]);
            combinationSum(candidates, target - candidates[i], i, path, result);
            path.remove(path.size() - 1);
        }
    }

    /** Combination Sum - another solution. */
    @tags.DFS
    @tags.Backtracking
    @tags.Array
    @tags.Company.Snapchat
    @tags.Company.Uber
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        if (candidates == null || candidates.length == 0) {
            return Collections.emptyList();
        }

        Arrays.sort(candidates);
        List<List<Integer>> result = new ArrayList<>();
        combinationSum2(candidates, 0, target, result,
                new ArrayList<Integer>());
        return result;
    }

    private void combinationSum2(int[] candidates, int index, int target,
            List<List<Integer>> result, List<Integer> path) {
        if (target == 0) {
            result.add(new ArrayList<>(path));
            return;
        } else if (target < 0 || index == candidates.length) {
            return;
        }

        path.add(candidates[index]);
        combinationSum2(candidates, index, target - candidates[index], result,
                path);
        path.remove(path.size() - 1);
        index++;
        while (index < candidates.length
                && candidates[index] == candidates[index - 1]) {
            index++;
        }
        combinationSum2(candidates, index, target, result, path);
    }

    /**
     * Combination Sum II.
     *
     * Given a collection of candidate numbers (C) and a target number (T), find
     * all unique combinations in C where the candidate numbers sums to T. Each
     * number in C may only be used once in the combination.
     *
     * Notice: All numbers (including target) will be positive integers.
     * Elements in a combination (a1, a2, ¡­ , ak) must be in non-descending
     * order. (ie, a1 ¡Ü a2 ¡Ü ¡­ ¡Ü ak). The solution set must not contain
     * duplicate combinations.
     *
     * Example: Given candidate set [10,1,6,7,2,1,5] and target 8, A solution
     * set is: [ [1,7], [1,2,5], [2,6], [1,1,6] ]
     *
     * @param num:
     *            Given the candidate numbers
     * @param target:
     *            Given the target number
     * @return: All the combinations that sum to target
     */
    @tags.DFS
    @tags.Backtracking
    @tags.Array
    @tags.Company.Snapchat
    public List<List<Integer>> combinationSumII(int[] num, int target) {
        if (num == null || num.length == 0) {
            return Collections.emptyList();
        }

        Arrays.sort(num);
        List<List<Integer>> result = new ArrayList<>();
        combinationSumII(num, target, 0, new ArrayList<Integer>(), result);
        return result;
    }

    private void combinationSumII(int[] num, int target, int pos,
            List<Integer> path, List<List<Integer>> result) {
        if (target == 0) {
            result.add(new ArrayList<>(path));
        } else if (target < 0) {
            return;
        }
        int prev = -1;
        for (int i = pos; i < num.length; i++) {
            if (num[i] != prev) {
                path.add(num[i]);
                combinationSumII(num, target - num[i], i + 1, path, result);
                path.remove(path.size() - 1);
                prev = num[i];
            }
        }
    }

    /** Combination Sum II - another solution */
    @tags.DFS
    @tags.Backtracking
    @tags.Array
    @tags.Company.Snapchat
    public List<List<Integer>> combinationSumII2(int[] num, int target) {
        if (num == null || num.length == 0) {
            return Collections.emptyList();
        }

        Arrays.sort(num);
        List<List<Integer>> result = new ArrayList<>();
        combinationSumII2(num, 0, target, result, new ArrayList<Integer>());
        return result;
    }

    private void combinationSumII2(int[] num, int index, int target,
            List<List<Integer>> result, List<Integer> path) {
        if (target == 0) {
            result.add(new ArrayList<>(path));
            return;
        } else if (index == num.length || target < 0) {
            return;
        }

        path.add(num[index]);
        combinationSumII2(num, index + 1, target - num[index], result, path);
        path.remove(path.size() - 1);
        index++;
        while (index < num.length && num[index] == num[index - 1]) {
            index++;
        }
        combinationSumII2(num, index, target, result, path);
    }

    /**
     * Combination Sum III.
     *
     * Find all possible combinations of k numbers that add up to a number n,
     * given that only numbers from 1 to 9 can be used and each combination
     * should be a unique set of numbers.
     *
     * Example 1: Input: k = 3, n = 7. Output: [[1,2,4]].
     *
     * Example 2: Input: k = 3, n = 9. Output: [[1,2,6], [1,3,5], [2,3,4]].
     *
     * @param k
     * @param n
     * @return
     */
    @tags.Array
    @tags.Backtracking
    @tags.Status.OK
    public List<List<Integer>> combinationSumIII(int k, int n) {
        List<List<Integer>> result = new ArrayList<>();
        combinationSumIII(k, n, 1, result, new ArrayList<Integer>());
        return result;
    }

    private void combinationSumIII(int k, int n, int num,
            List<List<Integer>> result, List<Integer> path) {
        if (n == 0 && k == 0) {
            result.add(new ArrayList<>(path));
            return;
        } else if (n < 0 || k < 0 || num == 10) {
            return;
        }

        path.add(num);
        combinationSumIII(k - 1, n - num, num + 1, result, path);
        path.remove(path.size() - 1);
        combinationSumIII(k, n, num + 1, result, path);
    }

    /**
     * Combination Sum IV.
     *
     * Given an integer array with all positive numbers and no duplicates, find
     * the number of possible combinations that add up to a positive integer
     * target.
     *
     * Example: nums = [1, 2, 3], target = 4. The possible combination ways are:
     * (1, 1, 1, 1) (1, 1, 2) (1, 2, 1) (1, 3) (2, 1, 1) (2, 2) (3, 1). Note
     * that different sequences are counted as different combinations. Therefore
     * the output is 7.
     *
     * Follow up: What if negative numbers are allowed in the given array? How
     * does it change the problem? What limitation we need to add to the
     * question to allow negative numbers?
     *
     * @param nums
     * @param target
     * @return
     */
    @tags.DynamicProgramming
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Company.Snapchat
    @tags.Status.NeedPractice
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int i = 1; i < dp.length; i++) {
            for (Integer num : nums) {
                if (i - num >= 0) {
                    dp[i] += dp[i - num];
                }
            }
        }
        return dp[target];
    }

    // ---------------------------------------------------------------------- //
    // ---------------------------- World Ladder ---------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Word Ladder.
     *
     * Given two words (start and end), and a dictionary, find the length of
     * shortest transformation sequence from start to end, such that: 1. Only
     * one letter can be changed at a time. 2. Each intermediate word must exist
     * in the dictionary.
     *
     * Notice: Return 0 if there is no such transformation sequence. All words
     * have the same length. All words contain only lowercase alphabetic
     * characters.
     *
     * Example: Given: start = "hit", end = "cog", dict = ["hot", "dot", "dog",
     * "lot", "log"]. As one shortest transformation is "hit" -> "hot" -> "dot"
     * -> "dog" -> "cog", return its length 5.
     *
     * @param start,
     *            a string
     * @param end,
     *            a string
     * @param dict,
     *            a set of string
     * @return an integer
     */
    @tags.BFS
    @tags.Company.Amazon
    @tags.Company.Facebook
    @tags.Company.LinkedIn
    @tags.Company.Snapchat
    @tags.Company.Yelp
    @tags.Status.SuperHard
    public int ladderLength(String start, String end, Set<String> dict) {
        if (start.equals(end)) {
            return 1;
        }

        Queue<String> queue = new LinkedList<>();
        queue.offer(start);
        Set<String> visited = new HashSet<>();
        visited.add(start);
        int len = 1;

        while (!queue.isEmpty()) {
            len++;
            Queue<String> level = new LinkedList<>();

            for (String word : queue) {
                char[] letters = word.toCharArray();
                for (int i = 0; i < letters.length; i++) {
                    char old = letters[i];
                    for (char c = 'a'; c <= 'z'; c++) {
                        if (c != old) {
                            letters[i] = c;
                            String newWord = new String(letters);

                            // check end
                            if (newWord.equals(end)) {
                                return len;
                            }
                            // check dict
                            if (dict.contains(newWord)
                                    && !visited.contains(newWord)) {
                                level.offer(newWord);
                                visited.add(newWord);
                            }
                        }
                    }
                    letters[i] = old;
                }
            }

            queue = level;
        }

        return 0;
    }

    /**
     * Word Ladder II
     *
     * Given two words (start and end), and a dictionary, find all shortest
     * transformation sequence(s) from start to end, such that:
     * Only one letter can be changed at a time.
     * Each intermediate word must exist in the dictionary.
     *
     * For example, Given:
     * start = "hit",
     * end = "cog",
     * dict = ["hot","dot","dog","lot","log"],
     * Return
     * [
     *  ["hit","hot","dot","dog","cog"],
     *  ["hit","hot","lot","log","cog"]
     * ].
     *
     * Note:
     * All words have the same length.
     * All words contain only lowercase alphabetic characters.
     *
     * @param start, a string
     * @param end, a string
     * @param dict, a set of string
     * @return a list of lists of string
     */
    @tags.BFS
    @tags.DFS
    @tags.Backtracking
    @tags.Array
    @tags.String
    @tags.Company.Amazon
    @tags.Company.Yelp
    @tags.Status.SuperHard
    public List<List<String>> findLadders(String start, String end,
            Set<String> dict) {
        Queue<String> queue = new LinkedList<>();
        queue.offer(start);
        Set<String> visited = new HashSet<>();
        visited.add(start);
        Map<String, List<String>> prev = new HashMap<>();

        // bfs to find all links from start to end
        while (!queue.isEmpty() && !prev.containsKey(end)) {
            Set<String> newVisited = new HashSet<>();

            // explore all possibilities for queued strings
            for (String word : queue) {
                char[] letters = word.toCharArray();

                for (int i = 0; i < letters.length; i++) {
                    char old = letters[i];
                    for (char c = 'a'; c <= 'z'; c++) {
                        if (c != old) {
                            letters[i] = c;
                            String newWord = new String(letters);

                            if (newWord.equals(end) || (dict.contains(newWord)
                                    && !visited.contains(newWord))) {
                                newVisited.add(newWord);
                                if (prev.containsKey(newWord)) {
                                    prev.get(newWord).add(word);
                                } else {
                                    prev.put(newWord, new ArrayList<String>());
                                    prev.get(newWord).add(word);
                                }

                            }
                        }
                    }
                    letters[i] = old;
                }
            }

            // offer the unvisited strings to next queue
            queue.clear();
            for (String newWord : newVisited) {
                if (!visited.contains(newWord)) {
                    queue.offer(newWord);
                    visited.add(newWord);
                }
            }
        }

        // no path found
        List<List<String>> ladders = new ArrayList<>();
        if (!prev.containsKey(end)) {
            return ladders;
        }

        // dfs to link all paths
        findLadders(end, ladders, new ArrayList<String>(), prev);
        return ladders;
    }

    private void findLadders(String s, List<List<String>> ladders,
            List<String> path, Map<String, List<String>> prev) {
        path.add(s);
        if (!prev.containsKey(s)) {
            List<String> list = new ArrayList<>(path);
            Collections.reverse(list);
            ladders.add(list);
        } else {
            for (String parent : prev.get(s)) {
                findLadders(parent, ladders, path, prev);
            }
        }
        path.remove(path.size() - 1);
    }

    // ---------------------------------------------------------------------- //
    // ------------------------------ Subsets ------------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Subsets.
     *
     * Given a set of distinct integers, return all possible subsets.
     *
     * Notice: Elements in a subset must be in non-descending order. The
     * solution set must not contain duplicate subsets.
     *
     * Example: If S = [1,2,3], a solution is: [ [3], [1], [2], [1,2,3], [1,3],
     * [2,3], [1,2], [] ]
     *
     * Challenge: Can you do it in both recursively and iteratively?
     *
     * @param S:
     *            A set of numbers.
     * @return: A list of lists. All valid subsets.
     */
    @tags.Recursion
    @tags.DFS
    @tags.Backtracking
    @tags.Company.Facebook
    @tags.Company.Uber
    @tags.Status.OK
    public ArrayList<ArrayList<Integer>> subsets(int[] nums) {
        Arrays.sort(nums);
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        subsets(nums, 0, result, new ArrayList<Integer>());
        return result;
    }

    private void subsets(int[] nums, int pos,
            ArrayList<ArrayList<Integer>> result, List<Integer> path) {
        if (pos == nums.length) {
            result.add(new ArrayList<>(path));
            return;
        }
        path.add(nums[pos]);
        subsets(nums, pos + 1, result, path);
        path.remove(path.size() - 1);
        subsets(nums, pos + 1, result, path);
    }

    /** Subsets - another DFS solution. */
    @tags.Recursion
    @tags.DFS
    @tags.Backtracking
    @tags.Company.Facebook
    @tags.Company.Uber
    public ArrayList<ArrayList<Integer>> subsets2(int[] S) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
        Arrays.sort(S); // not necessary
        ArrayList<Integer> tmp = new ArrayList<Integer>();
        res.add(tmp);
        subsets2(res, tmp, S, 0);
        return res;
    }

    private void subsets2(ArrayList<ArrayList<Integer>> res,
            ArrayList<Integer> tmp, int[] S, int pos) {
        for (int i = pos; i < S.length; i++) {
            tmp.add(S[i]);
            res.add(new ArrayList<Integer>(tmp));
            subsets2(res, tmp, S, i + 1);
            tmp.remove(tmp.size() - 1);
        }
    }

    /** Subsets - Iterative solution. */
    public ArrayList<ArrayList<Integer>> subsetsIter(int[] nums) {
        ArrayList<ArrayList<Integer>> subsets = new ArrayList<>();

        if (nums == null) {
            return subsets;
        }

        subsets.add(new ArrayList<Integer>());

        Arrays.sort(nums);

        for (Integer i : nums) {
            ArrayList<ArrayList<Integer>> newSets = new ArrayList<>();

            for (ArrayList<Integer> set : subsets) {
                ArrayList<Integer> newSet = new ArrayList<>(set);
                newSet.add(i);
                newSets.add(newSet);
            }

            subsets.addAll(newSets);
        }

        return subsets;
    }

    /**
     * Subsets II.
     *
     * Given a list of numbers that may has duplicate numbers, return all
     * possible subsets.
     *
     * Notice: Each element in a subset must be in non-descending order. The
     * ordering between two subsets is free. The solution set must not contain
     * duplicate subsets.
     *
     * Example: If S = [1,2,2], a solution is: [ [2], [1], [1,2,2], [2,2],
     * [1,2], [] ].
     *
     * Challenge: Can you do it in both recursively and iteratively?
     *
     * @param S:
     *            A set of numbers.
     * @return: A list of lists. All valid subsets.
     */
    @tags.Recursion
    @tags.DFS
    @tags.Backtracking
    public ArrayList<ArrayList<Integer>> subsetsWithDup(ArrayList<Integer> S) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        Collections.sort(S);
        result.add(new ArrayList<>(path));
        subsetsWithDup(S, 0, result, path);
        return result;
    }

    private void subsetsWithDup(ArrayList<Integer> S, int pos,
            ArrayList<ArrayList<Integer>> result, List<Integer> path) {
        for (int i = pos; i < S.size(); i++) {
            if (i == pos || S.get(i) != S.get(i - 1)) {
                path.add(S.get(i));
                result.add(new ArrayList<>(path));
                subsetsWithDup(S, i + 1, result, path);
                path.remove(path.size() - 1);
            }
        }
    }

    /** Subsets II - another DFS solution. */
    @tags.Recursion
    @tags.DFS
    @tags.Backtracking
    @tags.Status.NeedPractice
    public ArrayList<ArrayList<Integer>> subsetsWithDup2(ArrayList<Integer> S) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        Collections.sort(S);
        subsetsWithDup2(S, 0, result, new ArrayList<Integer>());
        return result;
    }

    private void subsetsWithDup2(ArrayList<Integer> S, int index,
            ArrayList<ArrayList<Integer>> result, ArrayList<Integer> path) {
        if (index == S.size()) {
            result.add(new ArrayList<>(path));
            return;
        }

        path.add(S.get(index));
        subsetsWithDup2(S, index + 1, result, path);
        index++;
        while (index < S.size() && S.get(index) == S.get(index - 1)) {
            index++;
        }
        path.remove(path.size() - 1);
        subsetsWithDup2(S, index, result, path);
    }

    /** Subsets II - Iterative solution. */
    @tags.Recursion
    @tags.DFS
    @tags.Backtracking
    public ArrayList<ArrayList<Integer>> subsetsWithDupIterative(
            ArrayList<Integer> S) {
        if (S == null) {
            return new ArrayList<>();
        }

        // Sort the list
        Collections.sort(S);

        ArrayList<ArrayList<Integer>> subsets = new ArrayList<>();
        subsets.add(new ArrayList<Integer>());

        for (int i = 0; i < S.size(); i++) {
            int num = S.get(i), count = 1;
            while (i + 1 < S.size() && S.get(i + 1) == num) {
                i++;
                count++;
            }

            ArrayList<ArrayList<Integer>> newSets = new ArrayList<>();
            for (ArrayList<Integer> oldSet : subsets) {
                int countDown = count;
                while (countDown-- > 0) {
                    ArrayList<Integer> newSet = new ArrayList<>(oldSet);
                    newSet.add(num);
                    newSets.add(newSet);
                    oldSet = new ArrayList<>(newSet);
                }
            }
            subsets.addAll(newSets);
        }

        return subsets;
    }

    // ---------------------------------------------------------------------- //
    // ---------------------------- Permutations ---------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Next Permutation.
     *
     * Given a list of integers, which denote a permutation. Find the next
     * permutation in ascending order.
     *
     * Notice: The list may contains duplicate integers.
     *
     * Example: For [1,3,2,3], the next permutation is [1,3,3,2] For [4,3,2,1],
     * the next permutation is [1,2,3,4]
     *
     * @param nums:
     *            an array of integers
     * @return: return nothing (void), do not return anything, modify nums
     *          in-place instead
     */
    @tags.Permutation
    @tags.Source.LintCode
    public int[] nextPermutation(int[] nums) {
        if (nums == null || nums.length < 2) {
            return nums;
        }

        // find the index before trailing descending sequence
        int n = nums.length;
        int index = -1;
        for (int i = n - 2; i >= 0; i--) {
            if (nums[i] < nums[i + 1]) {
                index = i;
                break;
            }
        }

        // copy the first part
        int[] next = new int[n];
        for (int i = 0; i <= index; i++) {
            next[i] = nums[i];
        }

        // reverse the trailing descending sequence
        for (int i = index + 1, j = n - 1; i < n; i++, j--) {
            next[i] = nums[j];
        }

        // swap the number at index and next bigger number
        if (index != -1) {
            for (int i = index + 1; i < n; i++) {
                if (next[i] > next[index]) {
                    int temp = next[index];
                    next[index] = next[i];
                    next[i] = temp;
                    break;
                }
            }
        }

        return next;
    }

    /**
     * Next Permutation II.
     *
     * Implement next permutation, which rearranges numbers into the
     * lexicographically next greater permutation of numbers.
     *
     * If such arrangement is not possible, it must rearrange it as the lowest
     * possible order (ie, sorted in ascending order).
     *
     * The replacement must be in-place, do not allocate extra memory.
     *
     * Here are some examples. Inputs are in the left-hand column and its
     * corresponding outputs are in the right-hand column. [1,2,3] ¡ú [1,3,2],
     * [3,2,1] ¡ú [1,2,3], [1,1,5] ¡ú [1,5,1].
     *
     * @param nums:
     *            an array of integers
     * @return: return nothing (void), do not return anything, modify nums
     *          in-place instead
     */
    @tags.Permutation
    @tags.Array
    public void nextPermutationII(int[] nums) {
        if (nums == null || nums.length < 2) {
            return;
        }

        // find the index before trailing descending sequence
        int n = nums.length;
        int index = -1;
        for (int i = n - 2; i >= 0; i--) {
            if (nums[i] < nums[i + 1]) {
                index = i;
                break;
            }
        }

        // reverse the desending sequence
        for (int i = index + 1, j = n - 1; i < j; i++, j--) {
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
        }

        // swap the number at index with the next larger number
        if (index != -1) {
            for (int i = index + 1; i < n; i++) {
                if (nums[index] < nums[i]) {
                    int temp = nums[i];
                    nums[i] = nums[index];
                    nums[index] = temp;
                    break;
                }
            }
        }
    }

    /**
     * Previous Permutation.
     *
     * Given a list of integers, which denote a permutation. Find the previous
     * permutation in ascending order.
     *
     * Notice: The list may contains duplicate integers.
     *
     * Example: For [1,3,2,3], the previous permutation is [1,2,3,3]. For
     * [1,2,3,4], the previous permutation is [4,3,2,1].
     *
     * @param nums:
     *            A list of integers
     * @return: A list of integers that's previous permuation
     */
    @tags.Permutation
    @tags.Source.LintCode
    public ArrayList<Integer> previousPermuation(ArrayList<Integer> nums) {
        if (nums == null) {
            return new ArrayList<>();
        } else if (nums.size() < 2) {
            return new ArrayList<>(nums);
        }

        // find the index before trailing acsending sequence
        int n = nums.size();
        int index = -1;
        for (int i = n - 2; i >= 0; i--) {
            if (nums.get(i) > nums.get(i + 1)) {
                index = i;
                break;
            }
        }

        // copy the first part
        ArrayList<Integer> prev = new ArrayList<>();
        for (int i = 0; i <= index; i++) {
            prev.add(nums.get(i));
        }

        // reverse the ascending sequence
        for (int i = n - 1; i > index; i--) {
            prev.add(nums.get(i));
        }

        // swap the number at index with next smaller number
        if (index != -1) {
            for (int i = index + 1; i < n; i++) {
                if (prev.get(i) < prev.get(index)) {
                    int temp = prev.get(index);
                    prev.set(index, prev.get(i));
                    prev.set(i, temp);
                    break;
                }
            }
        }

        return prev;
    }

    /**
     * Permutation Index.
     *
     * Given a permutation which contains no repeated number, find its index in
     * all the permutations of these numbers, which are ordered in
     * lexicographical order. The index begins at 1.
     *
     * Example: Given [1,2,4], return 1.
     *
     * @param A
     *            an integer array
     * @return a long integer
     */
    @tags.Permutation
    public long permutationIndex(int[] A) {
        List<Integer> list = new ArrayList<>();
        for (Integer i : A) {
            list.add(i);
        }
        Collections.sort(list);

        long factor = 1;
        for (int i = 2; i < A.length; i++) {
            factor *= i;
        }

        long countLess = 0;
        for (int i = 0, j = A.length - 1; i < A.length - 1; i++, j--) {
            int less = list.indexOf(A[i]);
            countLess += factor * less;
            factor /= j;
            list.remove(less);
        }

        return countLess + 1;
    }

    /**
     * Permutation Index II.
     *
     * Given a permutation which may contain repeated numbers, find its index in
     * all the permutations of these numbers, which are ordered in
     * lexicographical order. The index begins at 1.
     *
     * Example: Given the permutation [1, 4, 2, 2], return 3.
     *
     * @param A
     *            an integer array
     * @return a long integer
     */
    @tags.Permutation
    public long permutationIndexII(int[] A) {
        int n = A.length;
        List<Integer> sorted = new ArrayList<>();
        for (Integer i : A) {
            sorted.add(i);
        }
        Collections.sort(sorted);

        long factor = factorialL(n);
        long countLess = 0;
        for (int i = 0, j = n; i < n - 1; i++, j--) {
            int less = sorted.indexOf(A[i]);
            long newLess = factor / dupFactor(sorted);
            newLess = newLess * less / j;
            countLess += newLess;
            sorted.remove(less);
            factor /= j;
        }

        return countLess + 1;
    }

    private long dupFactor(List<Integer> sorted) {
        int prev = sorted.get(0);
        int count = 1;
        int dup = 1;
        for (int i = 1; i < sorted.size(); i++) {
            if (sorted.get(i) == prev) {
                count++;
            } else {
                dup *= factorialL(count);
                prev = sorted.get(i);
                count = 1;
            }
        }
        return dup * factorialL(count);
    }

    private long factorialL(int n) {
        long fact = 1;
        for (int i = n; i > 1; i--) {
            fact *= i;
        }
        return fact;
    }

    /**
     * Permutation Sequence.
     *
     * Given n and k, return the k-th permutation sequence.
     *
     * Notice: n will be between 1 and 9 inclusive.
     *
     * Example: For n = 3, all permutations are listed as follows: "123" "132"
     * "213" "231" "312" "321". If k = 4, the fourth permutation is "231".
     *
     * Challenge: O(n*k) in time complexity is easy, can you do it in O(n^2) or
     * less?
     *
     * @param n:
     *            n
     * @param k:
     *            the kth permutation
     * @return: return the k-th permutation
     */
    @tags.Permutation
    @tags.Array
    public String getPermutation(int n, int k) {
        List<Integer> list = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            list.add(i);
        }

        int all = factorial(n);

        StringBuilder sb = new StringBuilder();
        for (int i = 0, count = n; i < n; i++, count--) {
            int index = (k - 1) * count / all;
            sb.append(list.remove(index));
            all /= count;
            k -= index * all;
        }
        return sb.toString();
    }

    private int factorial(int n) {
        int fact = 1;
        for (int i = n; i > 1; i--) {
            fact *= i;
        }
        return fact;
    }

    /**
     * String Permutation II.
     *
     * Given a string, find all permutations of it without duplicates.
     *
     * Example: Given "abb", return ["abb", "bab", "bba"]. Given "aabb", return
     * ["aabb", "abab", "baba", "bbaa", "abba", "baab"].
     *
     * @param str
     *            a string
     * @return all permutations
     */
    @tags.DFS
    @tags.Backtracking
    @tags.String
    @tags.Recursion
    @tags.Permutation
    public List<String> stringPermutation2(String str) {
        char[] chars = str.toCharArray();
        Arrays.sort(chars);
        List<String> result = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        boolean[] picked = new boolean[chars.length];

        stringPermutation2(chars, picked, result, sb);
        return result;
    }

    private void stringPermutation2(char[] chars, boolean[] picked,
            List<String> result, StringBuilder sb) {
        if (sb.length() == chars.length) {
            result.add(sb.toString());
            return;
        }
        for (int i = 0; i < chars.length; i++) {
            if ((i > 0 && chars[i] == chars[i - 1] && !picked[i - 1])
                    || picked[i]) {
                continue;
            }
            sb.append(chars[i]);
            picked[i] = true;
            stringPermutation2(chars, picked, result, sb);
            sb.deleteCharAt(sb.length() - 1);
            picked[i] = false;
        }
    }

    /**
     * N-Queens.
     *
     * The n-queens puzzle is the problem of placing n queens on an n¡Án
     * chessboard such that no two queens attack each other. Given an integer n,
     * return all distinct solutions to the n-queens puzzle. Each solution
     * contains a distinct board configuration of the n-queens' placement, where
     * 'Q' and '.' both indicate a queen and an empty space respectively.
     *
     * Get all distinct N-Queen solutions.
     *
     * Example: There exist two distinct solutions to the 4-queens puzzle: [ //
     * Solution 1 [".Q..", "...Q", "Q...", "..Q." ], // Solution 2 ["..Q.",
     * "Q...", "...Q", ".Q.." ] ].
     *
     * Challenge: Can you do it without recursion?
     *
     * @param n:
     *            The number of queens
     * @return: All distinct solutions For example, A string '...Q' shows a
     *          queen on forth position
     */
    @tags.Recursion
    @tags.DFS
    @tags.Backtracking
    @tags.Status.NeedPractice
    public ArrayList<ArrayList<String>> solveNQueens(int n) {
        ArrayList<ArrayList<String>> result = new ArrayList<>();
        solveNQueens(n, result, new ArrayList<Integer>());
        return result;
    }

    private void solveNQueens(int n, ArrayList<ArrayList<String>> result,
            List<Integer> list) {
        if (list.size() == n) {
            result.add(translateNQueens(list));
            return;
        }

        for (int i = 0; i < n; i++) {
            if (isValid(list, i)) {
                list.add(i);
                solveNQueens(n, result, list);
                list.remove(list.size() - 1);
            }
        }
    }

    private boolean isValid(List<Integer> list, int newCol) {
        int newRow = list.size();
        for (int row = 0; row < newRow; row++) {
            int col = list.get(row);
            if (col == newCol || newRow - row == Math.abs(col - newCol)) {
                return false;
            }
        }
        return true;
    }

    private ArrayList<String> translateNQueens(List<Integer> list) {
        ArrayList<String> board = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < list.size(); i++) {
            sb.append('.');
        }
        for (int i = 0; i < list.size(); i++) {
            int index = list.get(i);
            sb.setCharAt(index, 'Q');
            board.add(sb.toString());
            sb.setCharAt(index, '.');
        }
        return board;
    }

    /**
     * N-Queens II.
     *
     * Follow up for N-Queens problem. Now, instead outputting board
     * configurations, return the total number of distinct solutions.
     *
     * Calculate the total number of distinct N-Queen solutions.
     *
     * Example: For n=4, there are 2 distinct solutions.
     *
     * @param n:
     *            The number of queens.
     * @return: The total number of distinct solutions.
     */
    @tags.Recursion
    @tags.DFS
    @tags.Backtracking
    @tags.Company.Zenefits
    @tags.Status.OK
    public int totalNQueens(int n) {
        List<Integer> list = new ArrayList<>();
        return totalNQueens(n, list);
    }

    private int totalNQueens(int n, List<Integer> list) {
        if (list.size() == n) {
            return 1;
        }

        int count = 0;
        for (int i = 0; i < n; i++) {
            if (isValid(list, i)) {
                list.add(i);
                count += totalNQueens(n, list);
                list.remove(list.size() - 1);
            }
        }
        return count;
    }

    /**
     * Permutations.
     *
     * Given a list of numbers, return all possible permutations.
     *
     * Notice: You can assume that there is no duplicate numbers in the list.
     *
     * Example: For nums = [1,2,3], the permutations are: [ [1,2,3], [1,3,2],
     * [2,1,3], [2,3,1], [3,1,2], [3,2,1] ].
     *
     * Challenge: Do it without recursion.
     *
     * @param nums:
     *            A list of integers.
     * @return: A list of permutations.
     */
    @tags.DFS
    @tags.Backtracking
    @tags.Recursion
    @tags.Permutation
    @tags.Company.LinkedIn
    @tags.Company.Microsoft
    @tags.Status.NeedPractice
    public ArrayList<ArrayList<Integer>> permute(ArrayList<Integer> nums) {
        if (nums == null) {
            return new ArrayList<>();
        }
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        ArrayList<Integer> path = new ArrayList<>();
        permute(nums, 0, result, path);
        return result;
    }

    private void permute(List<Integer> nums, int pos,
            ArrayList<ArrayList<Integer>> result, ArrayList<Integer> path) {
        if (pos == nums.size()) {
            result.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i <= pos; i++) {
            path.add(i, nums.get(pos));
            permute(nums, pos + 1, result, path);
            path.remove(i);
        }
    }

    /** Permutation - iterative solution. */
    @tags.Permutation
    @tags.Company.LinkedIn
    public ArrayList<ArrayList<Integer>> permute(int[] num) {
        ArrayList<ArrayList<Integer>> permutations = new ArrayList<ArrayList<Integer>>();

        if (num.length == 0) {
            return permutations;
        }

        // add a initial empty list
        permutations.add(new ArrayList<Integer>());

        // add one integer in original array each time
        for (Integer i : num) {
            // construct a new list to new generated permutations
            ArrayList<ArrayList<Integer>> update = new ArrayList<ArrayList<Integer>>();

            // add the integer to every old permutation
            for (ArrayList<Integer> permutation : permutations) {
                // add the new integer to any possible position
                for (int j = 0; j < permutation.size() + 1; j++) {
                    ArrayList<Integer> newPermutation = new ArrayList<Integer>();
                    newPermutation.addAll(permutation); // add existing elements
                    newPermutation.add(j, i); // add new integer at position j
                    update.add(newPermutation);
                }
            }

            // set the result to updated list of permutations
            permutations = update;
        }

        return permutations;
    }

    /**
     * Permutations II.
     *
     * Given a list of numbers with duplicate number in it. Find all unique
     * permutations.
     *
     * Example: For numbers [1,2,2] the unique permutations are: [ [1,2,2],
     * [2,1,2], [2,2,1] ].
     *
     * Challenge: Using recursion to do it is acceptable. If you can do it
     * without recursion, that would be great!
     *
     * @param nums:
     *            A list of integers.
     * @return: A list of unique permutations.
     */
    @tags.DFS
    @tags.Recursion
    @tags.Permutation
    @tags.Backtracking
    @tags.Company.LinkedIn
    @tags.Company.Microsoft
    @tags.Status.NeedPractice
    public ArrayList<ArrayList<Integer>> permuteUnique(
            ArrayList<Integer> nums) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (nums == null || nums.size() == 0) {
            return new ArrayList<>();
        }

        Collections.sort(nums);
        permuteUnique(nums, 0, result, new ArrayList<Integer>());
        return result;
    }

    private void permuteUnique(ArrayList<Integer> nums, int pos,
            ArrayList<ArrayList<Integer>> result, ArrayList<Integer> path) {
        if (pos == nums.size()) {
            result.add(new ArrayList<>(path));
            return;
        }

        int from = path.lastIndexOf(nums.get(pos)) + 1;
        for (int i = from; i <= path.size(); i++) {
            path.add(i, nums.get(pos));
            permuteUnique(nums, pos + 1, result, path);
            path.remove(i);
        }
    }

    /** Permutations II - My iterative solution. */
    @tags.Permutation
    @tags.Company.Microsoft
    @tags.Company.LinkedIn
    public ArrayList<ArrayList<Integer>> permuteUniqueIterative(int[] num) {
        ArrayList<ArrayList<Integer>> permutations = new ArrayList<ArrayList<Integer>>();

        if (num.length == 0) {
            return permutations;
        }

        permutations.add(new ArrayList<Integer>());

        // add one number to the permutations at one time
        for (Integer i : num) {
            ArrayList<ArrayList<Integer>> update = new ArrayList<ArrayList<Integer>>();

            for (ArrayList<Integer> permutation : permutations) {
                int from = permutation.lastIndexOf(i) + 1;
                for (int j = from; j < permutation.size() + 1; j++) {
                    ArrayList<Integer> newPermutation = new ArrayList<Integer>();
                    newPermutation.addAll(permutation);
                    newPermutation.add(j, i);
                    update.add(newPermutation);
                }
            }

            permutations = update;
        }

        return permutations;
    }

    // ---------------------------------------------------------------------- //
    // ----------------------------- UNION FIND ----------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Find the Connected Component in the Undirected Graph.
     *
     * Find the number connected component in the undirected graph. Each node in
     * the graph contains a label and a list of its neighbors. (a connected
     * component (or just component) of an undirected graph is a subgraph in
     * which any two vertices are connected to each other by paths, and which is
     * connected to no additional vertices in the supergraph.)
     *
     * Notice: Each connected component should sort by label.
     *
     * @param nodes
     *            a array of Undirected graph node
     * @return a connected set of a Undirected graph
     */
    @tags.UnionFind
    @tags.DFS
    @tags.BFS
    public List<List<Integer>> connectedSet(
            ArrayList<UndirectedGraphNode> nodes) {
        if (nodes == null || nodes.size() == 0) {
            return Collections.emptyList();
        }

        List<List<Integer>> result = new ArrayList<>();
        Set<UndirectedGraphNode> visited = new HashSet<>();

        for (UndirectedGraphNode node : nodes) {
            if (!visited.contains(node)) {
                Stack<UndirectedGraphNode> stack = new Stack<>();
                stack.push(node);
                List<Integer> union = new ArrayList<>();

                do {
                    UndirectedGraphNode current = stack.pop();
                    if (!visited.contains(current)) {
                        union.add(current.label);
                        visited.add(current);

                        for (UndirectedGraphNode neighbor : current.neighbors) {
                            stack.push(neighbor);
                        }
                    }
                } while (!stack.isEmpty());

                Collections.sort(union);
                result.add(union);
            }
        }

        return result;
    }

    /**
     * Find the Weak Connected Component in the Directed Graph.
     *
     * Find the number Weak Connected Component in the directed graph. Each node
     * in the graph contains a label and a list of its neighbors. (a connected
     * set of a directed graph is a subgraph in which any two vertices are
     * connected by direct edge path.)
     *
     * Notice: Sort the element in the set in increasing order
     *
     * @param nodes
     *            a array of Directed graph node
     * @return a connected set of a directed graph
     */
    @tags.UnionFind
    public List<List<Integer>> connectedSet2(
            ArrayList<DirectedGraphNode> nodes) {
        Map<DirectedGraphNode, DirectedGraphNode> leader = new HashMap<>();
        for (DirectedGraphNode node : nodes) {
            leader.put(node, node);
        }

        for (DirectedGraphNode node : nodes) {
            DirectedGraphNode nodeLeader = find(leader, node);
            for (DirectedGraphNode neighbor : node.neighbors) {
                // union the neighbor to node's group
                leader.put(find(leader, neighbor), nodeLeader);
            }
        }

        for (DirectedGraphNode node : nodes) {
            find(leader, node);
        }

        Map<DirectedGraphNode, List<Integer>> groups = new HashMap<>();
        for (DirectedGraphNode node : nodes) {
            DirectedGraphNode aLeader = leader.get(node);
            List<Integer> group = groups.get(aLeader);
            if (group == null) {
                group = new ArrayList<>();
                groups.put(aLeader, group);
            }
            group.add(node.label);
        }

        List<List<Integer>> result = new ArrayList<>();
        for (List<Integer> list : groups.values()) {
            Collections.sort(list);
            result.add(list);
        }

        return result;
    }

    private DirectedGraphNode find(
            Map<DirectedGraphNode, DirectedGraphNode> leader,
            DirectedGraphNode node) {
        if (leader.get(node) == node) {
            return node;
        }
        leader.put(node, find(leader, leader.get(node)));
        return leader.get(node);
    }

    /**
     * Graph Valid Tree.
     *
     * Given n nodes labeled from 0 to n - 1 and a list of undirected edges
     * (each edge is a pair of nodes), write a function to check whether these
     * edges make up a valid tree.
     *
     * Notice: You can assume that no duplicate edges will appear in edges.
     * Since all edges are undirected, [0, 1] is the same as [1, 0] and thus
     * will not appear together in edges.
     *
     * Example: Given n = 5 and edges = [[0, 1], [0, 2], [0, 3], [1, 4]], return
     * true. Given n = 5 and edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]],
     * return false.
     *
     * @param n
     *            an integer
     * @param edges
     *            a list of undirected edges
     * @return true if it's a valid tree, or false
     */
    @tags.UnionFind
    @tags.DFS
    @tags.BFS
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Company.Zenefits
    public boolean validTree(int n, int[][] edges) {
        if (edges.length != n - 1) {
            return false;
        }

        // init union-find
        int[] leader = new int[n];
        for (int i = 0; i < n; i++) {
            leader[i] = i;
        }

        for (int[] edge : edges) {
            if (find(leader, edge[0]) == find(leader, edge[1])) {
                return false;
            }

            // union
            leader[find(leader, edge[0])] = find(leader, edge[1]);
        }

        return true;
    }

    private int find(int[] leader, int n) {
        if (leader[n] == n) {
            return n;
        }

        leader[n] = find(leader, leader[n]);
        return leader[n];
    }

    /**
     * Surrounded Regions.
     *
     * Given a 2D board containing 'X' and 'O', capture all regions surrounded
     * by 'X'. A region is captured by flipping all 'O''s into 'X''s in that
     * surrounded region.
     *
     * Example: ["XXXX","XOOX","XXOX","XOXX"]. After capture all regions
     * surrounded by 'X', the board should be: ["XXXX","XXXX","XXXX","XOXX"].
     *
     * @param board
     *            a 2D board containing 'X' and 'O'
     * @return void
     */
    @tags.UnionFind
    @tags.DFS
    @tags.BFS
    public void surroundedRegions(char[][] board) {
        if (board == null || board.length <= 2 || board[0].length <= 2) {
            return;
        }
        int m = board.length, n = board[0].length;

        // first and last row
        for (int i = 0; i < n; i++) {
            traverseAndMark(board, 0, i);
            traverseAndMark(board, m - 1, i);
        }

        // first and last column
        for (int i = 0; i < m; i++) {
            traverseAndMark(board, i, 0);
            traverseAndMark(board, i, n - 1);
        }

        // capture 'O' and restore 'S'
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                } else if (board[i][j] == 'S') {
                    board[i][j] = 'O';
                }
            }
        }
    }

    private void traverseAndMark(char[][] board, int row, int col) {
        int m = board.length, n = board[0].length;
        if (row < 0 || row >= m || col < 0 || col >= n) {
            return;
        }

        char c = board[row][col];
        if (c == 'S' || c == 'X') {
            return;
        }

        board[row][col] = 'S';
        traverseAndMark(board, row - 1, col);
        traverseAndMark(board, row + 1, col);
        traverseAndMark(board, row, col - 1);
        traverseAndMark(board, row, col + 1);
    }

    // ---------------------------------------------------------------------- //
    // ----------------------------- UNIT TESTS ----------------------------- //
    // ---------------------------------------------------------------------- //

    @Test
    public void tests() {
        wordLaddersTests();
    }

    private void wordLaddersTests() {
        String start = "hot", end = "dog";
        Set<String> dict = new HashSet<>(Arrays.asList("hot","cog","dog","tot","hog","hop","pot","dot"));
        List<List<String>> result = findLadders(start, end, dict);
        Assert.assertEquals(Arrays.asList("hot","dot","dog"), result.get(1));
        Assert.assertEquals(Arrays.asList("hot","hog","dog"), result.get(0));
    }
}
