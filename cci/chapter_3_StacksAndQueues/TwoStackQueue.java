package chapter_3_StacksAndQueues;

import java.util.Stack;

/** 3.5 implement a Queue using two stacks */
public class TwoStackQueue<T> {
	private Stack<T> topNew;
	private Stack<T> topOld;

	public TwoStackQueue() {
		topNew = new Stack<T>();
		topOld = new Stack<T>();
	}

	/**
	 * Here's the trick. Move elements from topNew to topOld if the topOld is
	 * empty, just to make sure the topOld have the oldest element. It doesn't
	 * matter if we have elements on both stacks, because we just want to keep
	 * track of the oldest element, the position to insert the newest element
	 * and the correctness of the queue total sequence.
	 */
	private void shiftNewToOld() {
		if (topOld.isEmpty()) {
			while (!topNew.isEmpty()) {
				topOld.push(topNew.pop());
			}
		}
	}

	public int size() {
		return topNew.size() + topOld.size();
	}

	public void add(T element) {
		topNew.push(element);
	}

	public T peek() {
		shiftNewToOld();
		return topOld.peek();
	}

	public T remove() {
		shiftNewToOld();
		return topOld.pop();
	}

	public static void main(String[] args) {
		TwoStackQueue<Integer> queue = new TwoStackQueue<Integer>();
		queue.add(1);
		queue.add(2);
		queue.add(3);
		System.out.println("peek([1, 2, 3]): " + queue.peek());
		queue.remove();
		System.out.println("peek([2, 3]): " + queue.peek());
		queue.add(4);
		queue.add(5);
		System.out.println("peek([2, 3, 4, 5]): " + queue.peek());
	}

}
