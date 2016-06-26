package chapter_3_StacksAndQueues;

import java.util.ArrayList;
import java.util.EmptyStackException;
import java.util.Stack;

/** 3.3 compose stack as several stacks each with a same capacity */
public class SetOfStacks<T> {
	private int capacity;
	private ArrayList<Stack<T>> stacks;

	public SetOfStacks(int capacity) {
		this.capacity = capacity;
		stacks = new ArrayList<Stack<T>>();
		// TODO Auto-generated constructor stub
	}

	private Stack<T> getLastStack() {
		if (stacks.size() == 0) {
			return null;
		} else {
			return stacks.get(stacks.size() - 1);
		}
	}

	public void push(T value) {
		Stack<T> stack = getLastStack();
		// all stack is full
		if (stack == null || stack.size() == capacity) {
			Stack<T> newStack = new Stack<T>();
			stacks.add(newStack);
			newStack.push(value);
		} else {
			stack.push(value);
		}
	}

	/**
	 * The reason why not shifting all later stacks data is the trade off of
	 * time complexity, instead, just leave some middle stack not full.
	 * 
	 * @return
	 * @throws EmptyStackException
	 */
	public T pop() throws EmptyStackException {
		try {
			Stack<T> last = getLastStack();
			T value = last.pop();
			if (last.isEmpty()) {
				stacks.remove(stacks.size() - 1);
			}
			return value;
		} catch (NullPointerException e) {
			throw new EmptyStackException();
		}
	}

	public T popAt(int index) throws EmptyStackException {
		try {
			Stack<T> stack = stacks.get(index);
			T value = stack.pop();
			if (stack.size() == 0) {
				stacks.remove(index);
			}
			return value;
		} catch (IndexOutOfBoundsException e) {
			throw new EmptyStackException();
		}
	}

	public T peek() throws EmptyStackException {
		return getLastStack().peek();
	}

	public boolean isEmpty() {
		return getLastStack().isEmpty();
	}

	public void printStacks() {
		System.out.println("Printing stacks...");
		// print stack array
		for (int i = 0; i < stacks.size(); i++) {
			Stack<T> stack = stacks.get(i);
			for (int j = 0; j < stack.size(); j++) {
				System.out.print(stack.get(j) + " ");
			}
			System.out.println();
		}
	}

	public static void main(String[] args) {
		SetOfStacks<Integer> sos = new SetOfStacks<Integer>(3);

		sos.push(1);
		sos.printStacks();
		sos.push(2);
		sos.printStacks();
		sos.push(3);
		sos.printStacks();
		System.out.println("peek(): " + sos.peek() + "\n");

		sos.push(4);
		sos.printStacks();
		sos.push(5);
		sos.printStacks();
		sos.push(6);
		sos.printStacks();
		System.out.println("peek(): " + sos.peek() + "\n");

		sos.pop();
		sos.pop();
		System.out.println("pop() twice:");
		sos.printStacks();
		System.out.println();

		sos.popAt(0);
		System.out.println("popAt(0):");
		sos.printStacks();
		sos.popAt(0);
		System.out.println("popAt(0):");
		sos.printStacks();
		sos.popAt(0);
		System.out.println("popAt(0):");
		sos.printStacks();
		System.out.println();

		sos.pop();
		System.out.println("pop():");
		sos.printStacks();
	}

}
