package pastinterviews;


import java.util.*;

/**
 * Problem from Medallia
 * 
 * Guangcheng Lu
 * Andrew ID: guangchl
 *
 */
public class Medallia {
  
    /** Basic node class */
	private static class Node {
		private final int id;
		private final List<Node> children;
		
		Node(int id) {
			this.id = id;
			this.children = new ArrayList<Node>();
		}
		
		@Override public String toString() {
			return String.valueOf(id);
		}
	}
	
	private static Map<Integer, Integer> depthMap;
	private static Map<Integer, Integer> sizeMap;
    /** @return {@link List} of {@link Node}s which form the largest common subtrees */
	private static List<Node> getLargestCommonSubtrees(Node root) {
        // YOUR CODE HERE
		List<Node> result = new ArrayList<Node>();
		if (root == null) {
			return result;
		}
		
		Stack<ArrayList<Node>> allNodes = listNodes(root);
		ArrayList<Node> nodeList = new ArrayList<Node>();
		for (ArrayList<Node> list : allNodes) {
			nodeList.addAll(list);
		}
		
		depthMap = getDepthMap(allNodes);
		sizeMap = getSizeMap(allNodes);
		
		int maxSize = 0;
		
		for (int i = 0; i < allNodes.size(); i++) {
			for (int j = i; j < allNodes.size(); j++) {
				if (isCommon(nodeList.get(i), nodeList.get(j))) {
					int newSize = sizeMap.get(nodeList.get(i).id);
					if ( newSize > maxSize) {
						result.clear();
						result.add(nodeList.get(i));
						result.add(nodeList.get(j));
					} else if (newSize == maxSize) {
						result.add(nodeList.get(i));
						result.add(nodeList.get(j));
					}
				}
			}
		}
		
        return result;
	}
	
	private static Stack<ArrayList<Node>> listNodes(Node root) {
		Stack<ArrayList<Node>> stack = new Stack<ArrayList<Node>>();
		Queue<Node> queue = new LinkedList<Node>();
		
		queue.offer(root);
		
		while (!queue.isEmpty()) {
			Node node = queue.poll();
			ArrayList<Node> level = new ArrayList<Node>();
			
			for (Node child : node.children) {
				queue.offer(child);
				level.add(child);
			}
			
			stack.push(level);
		}
		
		return stack;
	}
	
	private static Map<Integer, Integer> getDepthMap(Stack<ArrayList<Node>> allNodes) {
		Map<Integer, Integer> depthMap = new HashMap<Integer, Integer>();
		
		for (int i = 0; i < allNodes.size(); i++) {
			ArrayList<Node> level = allNodes.get(i);
			for (Node node : level) {
				depthMap.put(node.id, i);
			}
		}
		
		return depthMap;
	}
	
	private static Map<Integer, Integer> getSizeMap(Stack<ArrayList<Node>> allNodes) {
		Map<Integer, Integer> sizeMap = new HashMap<Integer, Integer>();
		
		for (int i = 0; i < allNodes.size(); i++) {
			ArrayList<Node> level = allNodes.get(i);
			for (Node node : level) {
				if (node.children == null) {
					sizeMap.put(node.id, 0);
				} else {
					int size = 0;
					for (Node child : node.children) {
						size += sizeMap.get(child.id);
					}
					sizeMap.put(node.id, size);
				}
			}
		}
		
		return sizeMap;
	}
	
	private static boolean isCommon(Node node1, Node node2) {
		if (depthMap.get(node1.id) != depthMap.get(node2.id) 
				|| depthMap.get(node1.id) != depthMap.get(node2.id) 
				|| node1.id == node2.id 
				|| node1.children.size() != node2.children.size()) {
			return false;
		}
		
		for (int i = 0; i < node1.children.size(); i++) {
			if (!isCommon(node1.children.get(i), node2.children.get(i))) {
				return false;
			}
		}
		
		return true;
	}

	/** Useful for testing */
	private static void basicTest() {
		Node root = new Node(0);
		Node node1 = new Node(1);
		Node node2 = new Node(2);
		root.children.add(node1);
		root.children.add(node2);
		
		List<Node> result = getLargestCommonSubtrees(root);
		if (!result.contains(node1))
			throw new AssertionError(String.format("Expected to find node 1 but found nodes %s", getSortedOutput(result)));
		if (!result.contains(node2))
			throw new AssertionError(String.format("Expected to find node 2 but found nodes %s", getSortedOutput(result)));
	}

	public static String getSortedOutput(List<Node> result) {
		return "";
	}

	public static void main(String[] args) {
            basicTest();
    }
}
