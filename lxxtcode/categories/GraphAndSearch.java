package categories;

import java.util.ArrayList;
import java.util.HashMap;

public class GraphAndSearch {

    // -------------------------------- MODELS ---------------------------------

    /** Definition for undirected graph. */
    class UndirectedGraphNode {
        int label;
        ArrayList<UndirectedGraphNode> neighbors;

        UndirectedGraphNode(int x) {
            label = x;
            neighbors = new ArrayList<UndirectedGraphNode>();
        }
    }

    // ------------------------------- PROBLEMS --------------------------------

    // ---------------------------------- OLD ----------------------------------

    /**
     * Clone Graph
     *
     * If use iterative solution, copy node first, then copy connection use map
     */
    public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
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
	 * 5. N-Queens
	 */
    public ArrayList<String[]> solveNQueens(int n) {
        // IMPORTANT: Please reset any member data you declared, as
        // the same Solution instance will be reused for each test case.
        ArrayList<String[]> res = new ArrayList<String[]>();
        int[] loc = new int[n];
        dfs(res, loc, 0, n);
        return res;
    }

    private void dfs(ArrayList<String[]> res, int[] loc, int cur, int n) {
        if (cur == n)
            printboard(res, loc, n);
        else {
            for (int i = 0; i < n; i++) {
                loc[cur] = i;
                if (isValid(loc, cur))
                    dfs(res, loc, cur + 1, n);
            }
        }
    }

    private boolean isValid(int[] loc, int cur) {
        for (int i = 0; i < cur; i++) {
            if (loc[i] == loc[cur] || Math.abs(loc[i] - loc[cur]) == (cur - i))
                return false;
        }
        return true;
    }

    private void printboard(ArrayList<String[]> res, int[] loc, int n) {
        String[] ans = new String[n];
        for (int i = 0; i < n; i++) {
            String row = new String();
            for (int j = 0; j < n; j++) {
                if (j == loc[i])
                    row += "Q";
                else
                    row += ".";
            }
            ans[i] = row;
        }
        res.add(ans);
    }
}
