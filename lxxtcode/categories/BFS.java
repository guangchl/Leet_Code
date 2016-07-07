package categories;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;

public class BFS {

	/** Definition for binary tree */
	public class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;
		TreeNode(int x) { val = x; }
	}
	
	public ArrayList<ArrayList<Integer>> levelOrder(TreeNode root) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		
		Queue<ArrayList<TreeNode>> queue = new LinkedList<ArrayList<TreeNode>>();
		ArrayList<TreeNode> level = new ArrayList<TreeNode>();
		level.add(root);
		queue.offer(level);
		
		while (!queue.isEmpty()) {
			level = queue.poll();
			ArrayList<TreeNode> newLevel = new ArrayList<TreeNode>();
			ArrayList<Integer> intLevel = new ArrayList<Integer>();
			for (TreeNode node : level) {
				intLevel.add(node.val);
				if (node.left != null) {
					newLevel.add(node.left);
				}
				if (node.right != null) {
					newLevel.add(node.right);
				}
			}
			
			result.add(intLevel);
			if (newLevel.size() != 0) {
				queue.offer(newLevel);
			}
		}
		
		return result;
	}

}
