package pastinterviews;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
//import java.util.Scanner;

public class AutoRacer {

	public static class RacerTestInfo implements Comparable<RacerTestInfo> {
		public int id;
		public long start;
		public long end;
		
		public RacerTestInfo(int id, long start, long end) {
			this.id = id;
			this.start = start;
			this.end = end;
		}

		@Override
		public int compareTo(RacerTestInfo info) {
			if (start > info.start) {
				return 1;
			} else if (start < info.start) {
				return -1;
			} else {
				return 0;
			}
		}
	}
	
	public static class Node implements Comparable<Node> {
		public int id;
		public long end;
		public int count;
		
		public Node left;
		public Node right;
		
		public Node(int id, long end) {
			this.id = id;
			this.end = end;
		}

		@Override
		public int compareTo(Node node) {
			if (end > node.end) {
				return -1;
			} else if (end < node.end) {
				return 1;
			} else {
				return 0;
			}
		}
	}
	
	public static class Result implements Comparable<Result> {
		public int id;
		public int score;
		
		public Result(int id, int score) {
			this.id = id;
			this.score = score;
		}

		@Override
		public int compareTo(Result result) {
			return id - result.id;
		}
	}
	
	public static Node sortedArrayToBST(Node[] nodes) {
		if (nodes.length == 0) {
			return null;
		}
		return sortedArrayToBST(nodes, 0, nodes.length - 1);
	}

	public static Node sortedArrayToBST(Node[] nodes, int start, int end) {
		if (end == start) {
			nodes[start].count = 1;
			return nodes[start];
		} else if (end - start == 1) {
			nodes[start].count = 2;
			nodes[end].count = 1;
			nodes[start].right = nodes[end];
			return nodes[start];
		} else if (end - start == 2) {
			nodes[start + 1].count = 3;
			nodes[start].count = 1;
			nodes[end].count = 1;
			nodes[start].left = nodes[start];
			nodes[start].right = nodes[end];
			return nodes[start];
		}

		int mid = (start + end) / 2;
		nodes[mid].count = end - start + 1;
		nodes[mid].left = sortedArrayToBST(nodes, start, mid - 1);
		nodes[mid].right = sortedArrayToBST(nodes, mid + 1, end);

		return nodes[mid];
	}
	
	public static int search(Node root, long end) {
		root.count--;
		
		if (root.end == end) {
			return (root.left != null) ? root.left.count : 0;
		} else if (root.end < end) {
			int count = (root.left != null) ? root.left.count : 0;
			count++;
			if (root.right != null) {
				count += search(root.right, end);
			}
			return count;
		} else {
			if (root.left != null) {
				return search(root.left, end);
			} else {
				return 0;
			}
		}
	}
	
	public static void listScore(RacerTestInfo[] tests, Node[] nodes) {
		Arrays.sort(tests);
		Arrays.sort(nodes);
		
		Node root = sortedArrayToBST(nodes);
		
		Result[] results = new Result[nodes.length];
		for (int i = 0; i < nodes.length; i++) {
			results[i].id = nodes[i].id;
			results[i].score = search(root, nodes[i].end);
		}
	}

//	public static void main(String[] args) {
//		Scanner in = new Scanner(System.in);
//		int n = Integer.parseInt(in.nextLine());
//		RacerTestInfo[] tests = new RacerTestInfo[n];
//		Node[] nodes = new Node[n];
//		
//		for (int i = 0; in.hasNextLine(); i++) {
//			String[] line = in.nextLine().split(" ");
//			int id = Integer.parseInt(line[0]);
//			long start = Long.parseLong(line[1]);
//			long end = Long.parseLong(line[2]);
//			tests[i] = new RacerTestInfo(id, start, end);
//			nodes[i] = new Node(id, end);
//		}
//		in.close();
//		
//		listScore(tests, nodes);
//	}
	
	public static void main(String[] args) throws NumberFormatException, IOException {
		BufferedReader in = new BufferedReader(new FileReader(args[0]));
		int n = Integer.parseInt(in.readLine());
		RacerTestInfo[] tests = new RacerTestInfo[n];
		Node[] nodes = new Node[n];
		
		String oneLine = in.readLine();
		for (int i = 0; oneLine != null; i++) {
			String[] line = oneLine.split(" ");
			int id = Integer.parseInt(line[0]);
			long start = Long.parseLong(line[1]);
			long end = Long.parseLong(line[2]);
			tests[i] = new RacerTestInfo(id, start, end);
			nodes[i] = new Node(id, end);
		}
		in.close();
		
		listScore(tests, nodes);
	}
}
