package chapter_2_LinkedLists;

public class SimpleNode {
	int data;
	public SimpleNode next = null;
	public SimpleNode prev = null;
	
	public SimpleNode(int d) {
		data = d;
	}
	
	public void add(int d) {
		SimpleNode end = new SimpleNode(d);
        SimpleNode n = this;
        while (n.next != null) { n = n.next; }
        n.next = end;
	}
	
	public void addNode(SimpleNode node) {
		SimpleNode n = this;
        while (n.next != null) { n = n.next; }
        n.next = node;
	}
	
	public static void printList (SimpleNode head) {
		SimpleNode current;
		for (current = head; current != null; current = current.next) {
			System.out.print(current.data + " ");
		}
		System.out.println();
	}
}