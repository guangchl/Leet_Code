package ninechapter;

import java.util.ArrayList;
import java.util.Arrays;

public class DFS {

	/**
	 * 1. Subsets
	 * 
	 * Given a set of distinct integers, S, return all possible subsets.
	 * 
	 * Note: Elements in a subset must be in non-descending order. The solution
	 * set must not contain duplicate subsets.
	 */
    public ArrayList<ArrayList<Integer>> subsets(int[] S) {
		ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
		ArrayList<Integer> tmp = new ArrayList<Integer>();
		Arrays.sort(S);
		res.add(tmp);
		dfs(res, tmp, S, 0);
		return res;
	}

	public void dfs(ArrayList<ArrayList<Integer>> res, ArrayList<Integer> tmp,
			int[] S, int pos) {
		for (int i = pos; i <= S.length - 1; i++) {
			tmp.add(S[i]);
			res.add(new ArrayList<Integer>(tmp));
			dfs(res, tmp, S, i + 1);
			tmp.remove(tmp.size() - 1);
		}
    }
	
	/**
     * 2. Subsets II
     */
    public ArrayList<ArrayList<Integer>> subsetsWithDup(int[] num) {
		ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
		ArrayList<Integer> tmp = new ArrayList<Integer>();
		Arrays.sort(num);
		res.add(tmp);
		dfsWithDup(res, tmp, num, 0);
		return res;
	}

	public void dfsWithDup(ArrayList<ArrayList<Integer>> res,
			ArrayList<Integer> tmp, int[] num, int pos) {
		for (int i = pos; i <= num.length - 1; i++) {
			tmp.add(num[i]);
			res.add(new ArrayList<Integer>(tmp));
			dfsWithDup(res, tmp, num, i + 1);
			tmp.remove(tmp.size() - 1);
			// the only one line difference
			while (i < num.length - 1 && num[i] == num[i + 1]) i++;
		}
    }
	
	/**
	 * 3. Permutations
	 */
	public ArrayList<ArrayList<Integer>> permute(int[] num) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
        if (num == null || num.length == 0) {
            return result; 
        }

        ArrayList<Integer> list = new ArrayList<Integer>();
        helper(result, list, num);
        return result;
   }
   
   public void helper(ArrayList<ArrayList<Integer>> result, ArrayList<Integer> list, int[] num){
       if(list.size() == num.length) {
           result.add(new ArrayList<Integer>(list));
           return;
       }
       
       for(int i = 0; i<num.length; i++){
           if(list.contains(num[i])){
               continue;
           }
           list.add(num[i]);
           helper(result, list, num);
           list.remove(list.size() - 1);
       }
   }

	
	/**
	 * 4. Permutation II
	 */
	public ArrayList<ArrayList<Integer>> permuteUnique(int[] num) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (num == null || num.length == 0) {
			return result;
		}

		ArrayList<Integer> list = new ArrayList<Integer>();
		int[] visited = new int[num.length];
		Arrays.sort(num);
		permuteUniqueHelper(result, list, visited, num);

		return result;
	}

	public void permuteUniqueHelper(ArrayList<ArrayList<Integer>> result,
			ArrayList<Integer> list, int[] visited, int[] num) {
		if (list.size() == num.length) {
			result.add(new ArrayList<Integer>(list));
			return;
		}

		for (int i = 0; i < num.length; i++) {
			if (visited[i] == 1
					|| (i != 0 && num[i] == num[i - 1] && visited[i - 1] == 0)) {
				continue;
			}

			visited[i] = 1;
			list.add(num[i]);
			permuteUniqueHelper(result, list, visited, num);
			list.remove(list.size() - 1);
			visited[i] = 0;
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
        dfs(res,loc,0,n);
        return res;
    }  
    
    public void dfs(ArrayList<String[]> res, int[] loc, int cur, int n){  
        if(cur == n)   
            printboard(res, loc, n);  
        else{  
            for(int i = 0; i < n; i++){  
                loc[cur] = i;  
                if(isValid(loc, cur))  
                    dfs(res, loc, cur + 1,n);  
            }  
        }  
    }  
      
    public boolean isValid(int[] loc, int cur){  
        for(int i = 0; i < cur; i++){  
            if(loc[i] == loc[cur] || Math.abs(loc[i] - loc[cur]) == (cur - i))  
                return false;  
        }  
        return true;  
    }  
          
    public void printboard(ArrayList<String[]> res, int[] loc, int n){  
        String[] ans = new String[n];  
        for(int i = 0; i < n; i++){  
            String row = new String();  
            for(int j = 0; j < n; j++){
                if(j == loc[i]) 
                    row += "Q";  
                else row += ".";  
            }  
            ans[i] = row;  
        }  
        res.add(ans);
    }
}
