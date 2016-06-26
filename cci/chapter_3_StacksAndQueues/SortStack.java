package chapter_3_StacksAndQueues;

import java.util.Stack;

/** 3.6 sort a stack in ascending order (biggest items on top) */
/** you can use additional stacks but not other data structure like array */
public class SortStack {

	public static Stack<Integer> sort(Stack<Integer> stack) {
		
		if (stack.isEmpty()) {
			return stack;
		}
		
		Stack<Integer> buf = new Stack<Integer>();
		buf.push(stack.pop());
		while (!stack.isEmpty()) {
			int temp = stack.pop();
			while (!buf.isEmpty() && buf.peek() > temp) {
				stack.push(buf.pop());
			}
			buf.push(temp);
		}
		
		return buf;
		
		// if we must change the original stack, we have to push everything back
		//while (!buf.isEmpty()) {
		//	stack.push(buf.pop());
		//}
		// TODO Auto-generated constructor stub
	}

	public static void main(String[] args) {
		Stack<Integer> stack = new Stack<Integer>();

		stack.push(3);
		stack.push(4);
		stack.push(1);
		stack.push(2);
		stack.push(5);
		System.out.println(stack.toString());
		
		stack = SortStack.sort(stack);
		System.out.println(stack.toString());
	}

}
