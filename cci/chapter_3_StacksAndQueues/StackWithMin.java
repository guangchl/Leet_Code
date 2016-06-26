package chapter_3_StacksAndQueues;

import java.util.Stack;

/**
 * 3.2 stack with an additional function min which returns the minimum element.
 * Push, pop and min should all operate in O(1) time.
 */
public class StackWithMin extends Stack<Integer> {
	private static final long serialVersionUID = -8697983518216309238L;

	Stack<Integer> minimum;

	public StackWithMin() {
		super();
		minimum = new Stack<Integer>();
	}

	public void push(int value) {
		if (value <= min()) {
			minimum.push(value);
		}
		super.push(value);
	}

	public Integer pop() {
		int value = super.pop();
		if (value == min()) {
			minimum.pop();
		}
		return value;
	}

	/**
	 * @return minimum value in the stack
	 */
	public int min() {
		/* make sure the first push can be done */
		if (minimum.isEmpty()) {
			return Integer.MAX_VALUE;
		} else {
			return minimum.peek();
		}
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
