package clutter;

import java.util.ArrayList;


public class NoSuperLeafTreeCounter {
	
	public Object[] dp;
	
	public boolean strStr(String src, String dest) {
		if (src == null || dest == null) {
			return false;
		}
		
		int i, j;
		for (i = 0; i <= src.length() - dest.length(); i++) {
			for (j = 0; j < dest.length(); j++) {
				if (src.charAt(i + j) != dest.charAt(j)) {
					break;
				}
			}
			
			if (j == dest.length()) {
				return true;
			}
		}
		
		return false;
	}
	
	@SuppressWarnings("unchecked")
	public ArrayList<String> dfs(int n) {
		if (dp[n] == null) {
			dp[n] = new ArrayList<String>();
			
			for (int i = 0; i < n; i++) {
				for (String left : dfs(i)) {
					for (String right : dfs(n - i - 1)) {
						StringBuffer tree = new StringBuffer("(");
						tree.append(left);
						tree.append(right);
						tree.append(")");
						((ArrayList<String>)dp[n]).add(tree.toString());
					}
				}
			}
		}
		
		return (ArrayList<String>)dp[n];
	}
	
	@SuppressWarnings("unchecked")
	public void noSuperleaves(int N) {
		System.out.println("N = " + N);
		dp = new Object[N + 1];
		dp[0] = new ArrayList<String>();
		((ArrayList<String>)dp[0]).add("()");
		
		ArrayList<String> trees = dfs(N);
		System.out.println("total trees = " + trees.size());
		
		int result = 0;
		for (String tree : trees) {
			if (!strStr(tree, "((()())(()()))")) {
				result++;
			}
		}
		System.out.println("trees without superleaf = " + result);
	}
    
	public void test() {
		int N = 8;
		noSuperleaves(N);
	}
	
	public static void main(String[] args) {
		NoSuperLeafTreeCounter t = new NoSuperLeafTreeCounter();
		t.test();
	}
}
