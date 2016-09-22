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
     * Letter Combinations of a Phone Number.
     *
     * Given a digit string, return all possible letter combinations that the
     * number could represent. A mapping of digit to letters (just like on the
     * telephone buttons) is given below.
     *
     * Example: Input:Digit string "23", Output: ["ad", "ae", "af", "bd", "be",
     * "bf", "cd", "ce", "cf"].
     *
     * Note: Although the above answer is in lexicographical order, your answer
     * could be in any order you want.
     *
     * @param digits
     * @return
     */
    @tags.String
    @tags.Backtracking
    @tags.Company.Amazon
    @tags.Company.Dropbox
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Company.Uber
    @tags.Status.NeedPractice
    public List<String> letterCombinations(String digits) {
        List<String> result = new ArrayList<>();
        if (digits == null || digits.length() == 0) {
            return result;
        }

        // check illegal character
        int[] nums = new int[digits.length()];
        for (int i = 0; i < digits.length(); i++) {
            nums[i] = digits.charAt(i) - '0';
            if (nums[i] < 2 || nums[i] > 9) {
                return result;
            }
        }

        // digit to char mapping
        char[][] map = { { 'a', 'b', 'c' }, { 'd', 'e', 'f' },
                { 'g', 'h', 'i' }, { 'j', 'k', 'l' }, { 'm', 'n', 'o' },
                { 'p', 'q', 'r', 's' }, { 't', 'u', 'v' },
                { 'w', 'x', 'y', 'z' } };

        // dfs
        letterCombinations(nums, 0, map, new StringBuilder(), result);
        return result;
    }

    private void letterCombinations(int[] digits, int pos, char[][] map,
            StringBuilder path, List<String> result) {
        if (pos == digits.length) {
            result.add(path.toString());
            return;
        }

        int index = digits[pos] - 2;
        for (char c : map[index]) {
            path.append(c);
            letterCombinations(digits, pos + 1, map, path, result);
            path.deleteCharAt(path.length() - 1);
        }
    }

    /**
     * Walls and Gates.
     *
     * You are given a m x n 2D grid initialized with these three possible
     * values. -1 - A wall or an obstacle. 0 - A gate. INF - Infinity means an
     * empty room. We use the value 231 - 1 = 2147483647 to represent INF as you
     * may assume that the distance to a gate is less than 2147483647. Fill each
     * empty room with the distance to its nearest gate. If it is impossible to
     * reach a gate, it should be filled with INF.
     */
    @tags.BFS
    @tags.Company.Facebook
    @tags.Company.Google
    public void wallsAndGates(int[][] rooms) {
        if (rooms == null || rooms.length == 0 || rooms[0].length == 0) {
            return;
        }

        int m = rooms.length, n = rooms[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (rooms[i][j] == 0) {
                    bfs(rooms, i, j, 0);
                }
            }
        }
    }

    private void bfs(int[][] rooms, int i, int j, int dist) {
        int m = rooms.length, n = rooms[0].length;

        // index out of range
        if (i < 0 || i >= m || j < 0 || j >= n) {
            return;
        }

        // end of traversal
        if (rooms[i][j] == -1 || (rooms[i][j] <= dist && dist != 0)) {
            return;
        }

        rooms[i][j] = dist;
        dist++;

        // traverse neighbors (left, down, right, up)
        int[] xs = { 0, 1, 0, -1 };
        int[] ys = { -1, 0, 1, 0 };
        for (int k = 0; k < 4; k++) {
            bfs(rooms, i + xs[k], j + ys[k], dist);
        }
    }

    // ---------------------------------------------------------------------- //
    // ------------------------------ N-Queens ------------------------------ //
    // ---------------------------------------------------------------------- //

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
        // optimization 1: boolean array record visited column and 2 diagonals
        // optimization 2: bit manipulation instead of boolean array
        // optimization 3: enumerate all possible position without conflict
        // optimization 4: https://zhuanlan.zhihu.com/p/22846106?refer=maigo,
        // https://www.bittiger.io/classpage/jtDRXnKrncyioBLgs

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

    // ---------------------------------------------------------------------- //
    // ------------------------ Topological Sorting ------------------------- //
    // ---------------------------------------------------------------------- //

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
    @tags.Graph
    @tags.TopologicalSort
    @tags.BFS
    @tags.DFS
    @tags.Source.GeeksForGeeks
    @tags.Source.LintCode
    @tags.Status.Hard
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

    /**
     * Alien Dictionary.
     *
     * There is a new alien language which uses the latin alphabet. However, the
     * order among letters are unknown to you. You receive a list of words from
     * the dictionary, where words are sorted lexicographically by the rules of
     * this new language. Derive the order of letters in this language.
     *
     * For example, Given the following words in dictionary, { "wrt", "wrf",
     * "er", "ett", "rftt" ] The correct order is: "wertf".
     *
     * Note: You may assume all letters are in lowercase. If the order is
     * invalid, return an empty string. There may be multiple valid order of
     * letters, return any one of them is fine.
     *
     * @param words
     * @return
     */
    @tags.Graph
    @tags.TopologicalSort
    @tags.Company.Airbnb
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Company.PocketGems
    @tags.Company.Snapchat
    @tags.Company.Twitter
    @tags.Status.Hard
    public String alienOrder(String[] words) {
        Map<Character, Set<Character>> graph = new HashMap<>();
        Map<Character, Integer> refCount = new HashMap<>();

        // construct the graph
        for (int i = 1; i < words.length; i++) {
            String w1 = words[i - 1], w2 = words[i];
            int ptr1 = 0, ptr2 = 0;
            for (; ptr1 < w1.length() && ptr2 < w2.length(); ptr1++, ptr2++) {
                if (w1.charAt(ptr1) != w2.charAt(ptr2)) {
                    break;
                }
            }
            if (ptr1 < w1.length() && ptr2 < w2.length()) {
                char c1 = w1.charAt(ptr1), c2 = w2.charAt(ptr2);
                if (graph.containsKey(c1)) {
                    graph.get(c1).add(c2);
                } else {
                    Set<Character> set = new HashSet<>();
                    set.add(c2);
                    graph.put(c1, set);
                }
            }
        }

        // init ref count
        for (String word : words) {
            for (int i = 0; i < word.length(); i++) {
                refCount.put(word.charAt(i), 0);
            }
        }

        // calculate ref count
        for (Character c : graph.keySet()) {
            for (Character cc : graph.get(c)) {
                refCount.put(cc, refCount.get(cc) + 1);
            }
        }

        // topological sorting
        Queue<Character> queue = new LinkedList<>();
        for (Character c : refCount.keySet()) {
            if (refCount.get(c) == 0) {
                queue.offer(c);
            }
        }

        StringBuilder sb = new StringBuilder();
        while (!queue.isEmpty()) {
            char c = queue.poll();
            sb.append(c);
            if (graph.containsKey(c)) {
                for (Character neighbor : graph.get(c)) {
                    int count = refCount.get(neighbor) - 1;
                    if (count == 0) {
                        queue.offer(neighbor);
                    }
                    refCount.put(neighbor, count);
                }
            }
        }

        if (sb.length() < refCount.size()) {
            return "";
        }
        return sb.toString();
    }

    /**
     * Course Schedule.
     *
     * There are a total of n courses you have to take, labeled from 0 to n - 1.
     * Some courses may have prerequisites, for example to take course 0 you
     * have to first take course 1, which is expressed as a pair: [0,1] Given
     * the total number of courses and a list of prerequisite pairs, is it
     * possible for you to finish all courses?
     *
     * For example:
     *
     * 2, [[1,0]]. There are a total of 2 courses to take. To take course 1 you
     * should have finished course 0. So it is possible.
     *
     * 2, [[1,0],[0,1]]. There are a total of 2 courses to take. To take course
     * 1 you should have finished course 0, and to take course 0 you should also
     * have finished course 1. So it is impossible.
     *
     * Note: The input prerequisites is a graph represented by a list of edges,
     * not adjacency matrices. Read more about how a graph is represented.
     *
     * Hints: This problem is equivalent to finding if a cycle exists in a
     * directed graph. If a cycle exists, no topological ordering exists and
     * therefore it will be impossible to take all courses. Topological sort
     * could also be done via BFS.
     *
     * @param numCourses
     * @param prerequisites
     * @return
     */
    @tags.Graph
    @tags.TopologicalSort
    @tags.BFS
    @tags.DFS
    @tags.Company.Apple
    @tags.Company.Yelp
    @tags.Company.Zenefits
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        Map<Integer, Set<Integer>> graph = new HashMap<>();
        Map<Integer, Integer> refCount = new HashMap<>();

        // build the graph and record the ref count
        for (int[] pair : prerequisites) {
            int first = pair[1], then = pair[0];
            if (graph.containsKey(first)) {
                graph.get(first).add(then);
            } else {
                Set<Integer> set = new HashSet<>();
                set.add(then);
                graph.put(first, set);
            }

            // put every course in refCount
            refCount.put(first, 0);
            refCount.put(then, 0);
        }

        // calculate ref count
        for (Integer i : graph.keySet()) {
            for (Integer neighbor : graph.get(i)) {
                refCount.put(neighbor, refCount.get(neighbor) + 1);
            }
        }

        // not enough courses
        if (refCount.size() > numCourses) {
            return false;
        }

        // topological sort

        Queue<Integer> queue = new LinkedList<>();
        for (Integer i : refCount.keySet()) {
            if (refCount.get(i) == 0) {
                queue.offer(i);
            }
        }

        // count course can be traversed
        int count = 0;
        while (!queue.isEmpty()) {
            int course = queue.poll();
            count++;

            if (graph.containsKey(course)) {
                for (Integer i : graph.get(course)) {
                    refCount.put(i, refCount.get(i) - 1);
                    if (refCount.get(i) == 0) {
                        queue.offer(i);
                    }
                }
            }
        }

        return count == refCount.size();
    }

    /**
     * Course Schedule II.
     *
     * There are a total of n courses you have to take, labeled from 0 to n - 1.
     * Some courses may have prerequisites, for example to take course 0 you
     * have to first take course 1, which is expressed as a pair: [0,1]. Given
     * the total number of courses and a list of prerequisite pairs, return the
     * ordering of courses you should take to finish all courses. There may be
     * multiple correct orders, you just need to return one of them. If it is
     * impossible to finish all courses, return an empty array.
     *
     * For example: 2, [[1,0]]. There are a total of 2 courses to take. To take
     * course 1 you should have finished course 0. So the correct course order
     * is [0,1]. 4, [[1,0],[2,0],[3,1],[3,2]]. There are a total of 4 courses to
     * take. To take course 3 you should have finished both courses 1 and 2.
     * Both courses 1 and 2 should be taken after you finished course 0. So one
     * correct course order is [0,1,2,3]. Another correct ordering is[0,2,1,3].
     *
     * Note: The input prerequisites is a graph represented by a list of edges,
     * not adjacency matrices. Read more about how a graph is represented.
     *
     * Hints: This problem is equivalent to finding the topological order in a
     * directed graph. If a cycle exists, no topological ordering exists and
     * therefore it will be impossible to take all courses. Topological Sort via
     * DFS - A great video tutorial (21 minutes) on Coursera explaining the
     * basic concepts of Topological Sort. Topological sort could also be done
     * via BFS.
     */
    @tags.Graph
    @tags.TopologicalSort
    @tags.BFS
    @tags.DFS
    @tags.Company.Facebook
    @tags.Company.Zenefits
    @tags.Status.NeedPractice
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        // directed graph need both graph and refCount
        // while undirected graph only need graph since link is double sided
        Map<Integer, Set<Integer>> graph = new HashMap<>();
        Map<Integer, Integer> refCount = new HashMap<>();
        for (int i = 0; i < numCourses; i++) {
            graph.put(i, new HashSet<Integer>());
            refCount.put(i, 0);
        }

        // construct the graph
        for (int[] pair : prerequisites) {
            graph.get(pair[1]).add(pair[0]);
        }

        // calculate ref count
        for (Integer node : graph.keySet()) {
            for (Integer next : graph.get(node)) {
                refCount.put(next, refCount.get(next) + 1);
            }
        }

        Queue<Integer> queue = new LinkedList<>();
        for (Integer node : refCount.keySet()) {
            if (refCount.get(node) == 0) {
                queue.offer(node);
            }
        }

        int[] result = new int[numCourses];
        int index = 0;
        while (!queue.isEmpty()) {
            int node = queue.poll();
            result[index++] = node;

            for (Integer next : graph.get(node)) {
                if (refCount.get(next) == 1) {
                    queue.offer(next);
                }
                refCount.put(next, refCount.get(next) - 1);
            }
        }

        if (index != numCourses) {
            return new int[0];
        }
        return result;
    }

    /**
     * Minimum Height Trees.
     *
     * For a undirected graph with tree characteristics, we can choose any node
     * as the root. The result graph is then a rooted tree. Among all possible
     * rooted trees, those with minimum height are called minimum height trees
     * (MHTs). Given such a graph, write a function to find all the MHTs and
     * return a list of their root labels.
     *
     * Format: The graph contains n nodes which are labeled from 0 to n - 1. You
     * will be given the number n and a list of undirected edges (each edge is a
     * pair of labels). You can assume that no duplicate edges will appear in
     * edges. Since all edges are undirected, [0, 1] is the same as [1, 0] and
     * thus will not appear together in edges.
     *
     * Example 1: Given n = 4, edges = [[1, 0], [1, 2], [1, 3]], return [1].
     *
     * Example 2: Given n = 6, edges = [[0, 3], [1, 3], [2, 3], [4, 3], [5, 4]],
     * return [3, 4].
     *
     * @param n
     * @param edges
     * @return
     */
    @tags.Graph
    @tags.BFS
    @tags.Company.Google
    @tags.Status.Hard
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        if (n == 1) {
            Integer[] single = { 0 };
            return Arrays.asList(single);
        }

        // directed graph need both graph and refCount
        // while undirected graph only need graph since link is double sided

        // construct the graph
        Map<Integer, Set<Integer>> graph = new HashMap<>();
        for (int i = 0; i < n; i++) {
            graph.put(i, new HashSet<Integer>());
        }
        for (int[] edge : edges) {
            graph.get(edge[0]).add(edge[1]);
            graph.get(edge[1]).add(edge[0]);
        }

        Set<Integer> level = new HashSet<>();
        // find the leaves
        for (Integer node : graph.keySet()) {
            if (graph.get(node).size() == 1) {
                level.add(node);
            }
        }

        // BFS to find the longest path
        while (true) {
            Set<Integer> next = new HashSet<>();

            // add all neighbors with 2 links to next level
            for (Integer node : level) {
                for (Integer neighbor : graph.get(node)) {
                    if (graph.get(neighbor).size() == 2) {
                        next.add(neighbor);
                    }
                    graph.get(neighbor).remove(node);
                }
                graph.remove(node);
            }

            // return if no new level
            if (next.size() == 0) {
                return new ArrayList<>(level);
            }

            level = next;
        }
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
    @tags.Status.Hard
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

                    // back tracking
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
    @tags.Status.Hard
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
     * String Permutation.
     *
     * Given two strings, write a method to decide if one is a permutation of
     * the other.
     *
     * Example: abcd is a permutation of bcad, but abbe is not a permutation of
     * abe.
     *
     * @param A a string
     * @param B a string
     * @return a boolean
     */
    @tags.String
    @tags.Permutation
    @tags.Status.NeedPractice
    public boolean stringPermutation(String A, String B) {
        if (A == null && B == null) {
            return true;
        } else if (A == null || B == null) {
            return false;
        }

        int len = A.length();
        if (len != B.length()) {
            return false;
        }

        Map<Character, Integer> letters = new HashMap<>();
        for (int i = 0; i < len; i++) {
            char c = A.charAt(i);
            if (letters.containsKey(c)) {
                letters.put(c, letters.get(c) + 1);
            } else {
                letters.put(c, 1);
            }
        }

        for (int i = 0; i < len; i++) {
            char c = B.charAt(i);
            if (!letters.containsKey(c) || letters.get(c) == 0) {
                return false;
            }
            letters.put(c, letters.get(c) - 1);
        }

        return true;
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
    @tags.Company.Microsoft
    @tags.Status.OK
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
    @tags.Status.OK
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
                    ArrayList<Integer> newPermutation = new ArrayList<Integer>(
                            permutation);
                    newPermutation.add(j, i);
                    update.add(newPermutation);
                }
            }

            permutations = update;
        }

        return permutations;
    }

    /**
     * Generate Parentheses.
     *
     * Given n pairs of parentheses, write a function to generate all
     * combinations of well-formed parentheses.
     *
     * For example, given n = 3, a solution set is:
     * "((()))", "(()())", "(())()", "()(())", "()()()"
     */
    @tags.String
    @tags.Recursion
    @tags.Backtracking
    @tags.Company.Google
    @tags.Company.Zenefits
    @tags.Status.NeedPractice
    public ArrayList<String> generateParenthesis(int n) {
        ArrayList<String> result = new ArrayList<String>();
        StringBuffer sb = new StringBuffer();
        parenthesisRecursive(n, n, sb, result);
        return result;
    }

    public void parenthesisRecursive(int openStock, int closeStock,
            StringBuffer sb, ArrayList<String> result) {
        // if no "(" and ")" left, done with one combination
        if (openStock == 0 && closeStock == 0) {
            result.add(sb.toString());
            return;
        }

        // if still have "(" in stock
        if (openStock > 0) {
            sb.append("(");
            parenthesisRecursive(openStock - 1, closeStock, sb, result);
            sb.deleteCharAt(sb.length() - 1);
        }

        // if still have ")" in stock and in a valid position
        if (closeStock > openStock) {
            sb.append(")");
            parenthesisRecursive(openStock, closeStock - 1, sb, result);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    /**
     * Remove Invalid Parentheses.
     *
     * Remove the minimum number of invalid parentheses in order to make the
     * input string valid. Return all possible results.
     *
     * Note: The input string may contain letters other than the parentheses (
     * and ).
     *
     * Examples:
     *
     * "()())()" -> ["()()()", "(())()"]
     *
     * "(a)())()" -> ["(a)()()", "(a())()"]
     *
     * ")(" -> [""]
     */
    @tags.BFS
    @tags.DFS
    @tags.Company.Facebook
    @tags.Status.Hard
    public List<String> removeInvalidParentheses(String s) {
        List<String> result = new ArrayList<>();
        Set<String> visited = new HashSet<>();
        Queue<String> queue = new LinkedList<>();
        queue.offer(s);
        visited.add(s);

        // BFS
        while (!queue.isEmpty()) {
            Queue<String> next = new LinkedList<>();

            while (!queue.isEmpty()) {
                String p = queue.poll();

                if (isValid(p)) {
                    result.add(p);
                } else {
                    for (int i = 0; i < p.length(); i++) {
                        if (p.charAt(i) == '(' || p.charAt(i) == ')') {
                            // remove character at i, add to next level
                            String q = p.substring(0, i)
                                    + p.substring(i + 1, p.length());
                            if (!visited.contains(q)) {
                                visited.add(q);
                                next.offer(q);
                            }
                        }
                    }
                }
            }

            if (!result.isEmpty()) {
                return result;
            }

            queue = next;
        }

        return result;
    }

    private boolean isValid(String s) {
        int count = 0;
        for (char c : s.toCharArray()) {
            if (c == '(') {
                count++;
            } else if (c == ')') {
                count--;
            }

            if (count < 0) {
                return false;
            }
        }

        return count == 0;
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

    /**
     * Number of Islands.
     *
     * Given a boolean 2D matrix, find the number of islands.
     *
     * Notice: 0 is represented as the sea, 1 is represented as the island. If
     * two 1 is adjacent, we consider them in the same island. We only consider
     * up/down/left/right adjacent.
     *
     * Example: Given graph: [ [1, 1, 0, 0, 0], [0, 1, 0, 0, 1], [0, 0, 0, 1,
     * 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1] ] return 3.
     *
     * @param grid
     *            a boolean 2D matrix
     * @return an integer
     */
    @tags.DFS
    @tags.BFS
    @tags.UnionFind
    @tags.Company.Amazon
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Company.Microsoft
    @tags.Company.Zenefits
    @tags.Status.NeedPractice
    public int numIslands(boolean[][] grid) {
        // time: O(mn), since you will only traverse each tile 4 times

        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }

        int m = grid.length, n = grid[0].length;
        int count = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j]) {
                    dfs(grid, m, n, i, j);
                    count++;
                }
            }
        }

        return count;
    }

    private void dfs(boolean[][] grid, int m, int n, int x, int y) {
        if (x < 0 || x >= m || y < 0 || y >= n || !grid[x][y]) {
            return;
        }

        grid[x][y] = false;

        dfs(grid, m, n, x - 1, y);
        dfs(grid, m, n, x + 1, y);
        dfs(grid, m, n, x, y - 1);
        dfs(grid, m, n, x, y + 1);
    }

    /** Number of Islands - Union find. */
    @tags.DFS
    @tags.BFS
    @tags.UnionFind
    @tags.Company.Amazon
    @tags.Company.Facebook
    @tags.Company.Google
    @tags.Company.Microsoft
    @tags.Company.Zenefits
    @tags.Status.NeedPractice
    public int numIslands(char[][] grid) {
        // TODO find the big O complexity
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }

        int m = grid.length, n = grid[0].length;
        int[] leader = new int[m * n];

        // traverse and perform union find
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int index = i * n + j;
                if (grid[i][j] == '1') {
                    leader[index] = index;
                    int[] is = { 0, -1 };
                    int[] js = { -1, 0 };
                    for (int k = 0; k < 2; k++) {
                        int x = i + is[k], y = j + js[k];
                        if (x >= 0 && x < m && y >= 0 && y < n
                                && grid[x][y] == '1') {
                            int xy = x * n + y;
                            leader[leader[index]] = find(leader, xy);
                        }
                    }
                } else {
                    leader[index] = -1;
                }
            }
        }

        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < m * n; i++) {
            if (leader[i] != -1) {
                leader[i] = find(leader, i);
                set.add(leader[i]);
            }
        }
        return set.size();
    }

    /**
     * Number of Islands II.
     *
     * Given a n,m which means the row and column of the 2D matrix and an array
     * of pair A( size k). Originally, the 2D matrix is all 0 which means there
     * is only sea in the matrix. The list pair has k operator and each operator
     * has two integer A[i].x, A[i].y means that you can change the grid
     * matrix[A[i].x][A[i].y] from sea to island. Return how many island are
     * there in the matrix after each operator.
     *
     * Notice: 0 is represented as the sea, 1 is represented as the island. If
     * two 1 is adjacent, we consider them in the same island. We only consider
     * up/down/left/right adjacent.
     *
     * Example: Given n = 3, m = 3, array of pair A = [(0,0),(0,1),(2,2),(2,1)].
     * return [1,1,2,2].
     *
     * Challenge: Can you do it in time complexity O(k log mn), where k is the
     * length of the positions.
     *
     * @param n
     *            an integer
     * @param m
     *            an integer
     * @param operators
     *            an array of point
     * @return an integer array
     */
    @tags.UnionFind
    @tags.Company.Google
    @tags.Status.NeedPractice
    public List<Integer> numIslands2(int n, int m, Point[] operators) {
        // TODO find the big O complexity

        List<Integer> result = new ArrayList<>();
        if (n < 1 || m < 1 || operators == null || operators.length < 1) {
            return result;
        }

        int[] leader = new int[n * m];
        for (int i = 0; i < n * m; i++) {
            leader[i] = -1;
        }
        int count = 0;

        for (Point p : operators) {
            int index = p.x * m + p.y;
            if (leader[index] == -1) {
                count++;
                leader[index] = index;
                int[] is = {0, -1, 0, 1};
                int[] js = {-1, 0, 1, 0};
                for (int k = 0; k < 4; k++) {
                    int i = p.x + is[k], j = p.y + js[k];
                    int neighbor = i * m + j;
                    if (i >= 0 && i < n && j >= 0 && j < m && leader[neighbor] != -1) {
                        // merge neighbor to the point, make this point leader
                        int nleader = find(leader, neighbor);
                        if (nleader != index) {
                            leader[nleader] = index;
                            count--;
                        }
                    }
                }
            }
            result.add(count);
        }
        return result;
    }

    // ---------------------------------------------------------------------- //
    // ----------------------------- UNIT TESTS ----------------------------- //
    // ---------------------------------------------------------------------- //

    @Test
    public void tests() {
        canFinishTests();
        wordLaddersTests();
    }

    private void canFinishTests() {
        int numCourses = 10;
        int[][] prerequisites = {{5,8},{3,5},{1,9},{4,5},{0,2},{1,9},{7,8},{4,9}};
        Assert.assertTrue(canFinish(numCourses, prerequisites));

        int numCourses2 = 4;
        int[][] prerequisites2 = {{0,1},{3,1},{1,3},{3,2}};
        Assert.assertFalse(canFinish(numCourses2, prerequisites2));
    }

    private void wordLaddersTests() {
        String start = "hot", end = "dog";
        Set<String> dict = new HashSet<>(Arrays.asList("hot","cog","dog","tot","hog","hop","pot","dot"));
        List<List<String>> result = findLadders(start, end, dict);
        Assert.assertEquals(Arrays.asList("hot","dot","dog"), result.get(1));
        Assert.assertEquals(Arrays.asList("hot","hog","dog"), result.get(0));
    }
}
