package ninechapter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class DFS {

	/**
	 * 1. Subsets.
	 *
	 * Given a set of distinct integers, S, return all possible subsets.
	 *
	 * Note: Elements in a subset must be in non-descending order. The solution
	 * set must not contain duplicate subsets.
	 */
    public ArrayList<ArrayList<Integer>> subsets(int[] S) {
		ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
		Arrays.sort(S); // not necessary
		ArrayList<Integer> tmp = new ArrayList<Integer>();
		res.add(tmp);
		dfsSubset(res, tmp, S, 0);
		return res;
	}

	private void dfsSubset(ArrayList<ArrayList<Integer>> res, ArrayList<Integer> tmp,
			int[] S, int pos) {
		for (int i = pos; i < S.length; i++) {
			tmp.add(S[i]);
			res.add(new ArrayList<Integer>(tmp));
			dfsSubset(res, tmp, S, i + 1);
			tmp.remove(tmp.size() - 1);
		}
    }

	/**
	 * Subsets.
	 *
	 * Iterative solution.
	 *
     * @param S: A set of numbers.
     * @return: A list of lists. All valid subsets.
     */
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
     * 2. Subsets II
     *
	 * Given a list of numbers that may has duplicate numbers, return all
	 * possible subsets.
	 *
	 * Notice: Each element in a subset must be in non-descending order. The
	 * ordering between two subsets is free. The solution set must not contain
	 * duplicate subsets.
     */
    public ArrayList<ArrayList<Integer>> subsetsWithDup(int[] num) {
		ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
		Arrays.sort(num);
		ArrayList<Integer> tmp = new ArrayList<Integer>();
		res.add(tmp);
		dfsSubsetWithDup(res, tmp, num, 0);
		return res;
	}

	private void dfsSubsetWithDup(ArrayList<ArrayList<Integer>> res,
			ArrayList<Integer> tmp, int[] num, int pos) {
		for (int i = pos; i < num.length; i++) {
			tmp.add(num[i]);
			res.add(new ArrayList<Integer>(tmp));
			dfsSubsetWithDup(res, tmp, num, i + 1);
			tmp.remove(tmp.size() - 1);
			// the only one line difference
			while (i < num.length - 1 && num[i] == num[i + 1]) i++;
		}
    }

	/**
	 * Subsets II.
	 *
	 * Iterative solution.
	 *
	 * @param S:
	 *            A set of numbers.
	 * @return: A list of lists. All valid subsets.
	 */
    public ArrayList<ArrayList<Integer>> subsetsWithDupIterative(ArrayList<Integer> S) {
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

	/**
	 * 3. Permutations
	 *
	 * Given a list of numbers, return all possible permutations.
	 */
    public ArrayList<ArrayList<Integer>> permute(ArrayList<Integer> nums) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (nums == null) {
            return result;
        }

        ArrayList<Integer> temp = new ArrayList<>();
        dfsPermute(result, temp, nums, 0);

        return result;
    }

    private void dfsPermute(ArrayList<ArrayList<Integer>> result, ArrayList<Integer> temp,
             ArrayList<Integer> nums, int index) {
        if (index == nums.size()) {
            result.add(new ArrayList<>(temp));
            return;
        }

        for (int i = 0; i <= temp.size(); i++) {
            temp.add(i, nums.get(index));
            dfsPermute(result, temp, nums, index + 1);
            temp.remove(i);
        }
    }

	/**
	 * Permutation.
	 *
	 * My iterative solution.
	 */
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
	 * 4. Permutation II.
	 *
	 * Given a list of numbers with duplicate number in it. Find all unique
	 * permutations.
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

	private void permuteUniqueHelper(ArrayList<ArrayList<Integer>> result,
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
	 * Permutation II
	 *
	 * My dfs solution.
	 *
     * @param nums: A list of integers.
     * @return: A list of unique permutations.
     */
    public ArrayList<ArrayList<Integer>> permuteUnique(ArrayList<Integer> nums) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (nums == null) {
            return result;
        }

        ArrayList<Integer> temp = new ArrayList<>();
        dfs(result, temp, nums, 0);

        return result;
    }

    private void dfs(ArrayList<ArrayList<Integer>> result,
                     ArrayList<Integer> temp, ArrayList<Integer> nums,
                     int pos) {
        if (pos == nums.size()) {
            result.add(new ArrayList<>(temp));
            return;
        }

        int from = temp.lastIndexOf(nums.get(pos)) + 1;
        for (int i = from; i <= temp.size(); i++) {
            temp.add(i, nums.get(pos));
            dfs(result, temp, nums, pos + 1);
            temp.remove(i);
        }
    }

    /**
	 * Permutations II
	 *
	 * My iterative solution.
	 */
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
