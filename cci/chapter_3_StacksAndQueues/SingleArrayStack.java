package chapter_3_StacksAndQueues;

import java.util.EmptyStackException;

/** 3.1 use a single array to implement 3 stacks */
/** here in my implementation, 3 can be any positive integer */
public class SingleArrayStack {
	private int[] stackArr;
	private int[] stackTop;
	private int[] size;

	public SingleArrayStack(int arrSize, int stackNum) {
		stackArr = new int[arrSize];
		stackTop = new int[stackNum];
		for (int i = 0; i < stackTop.length; i++) {
			stackTop[i] = -1;
		}
		size = new int[stackNum];
	}

	private int getStackSize(int stackNum) {
		return size[stackNum];
	}

	private int getTotalSize() {
		int sum = 0;
		for (int i : size) {
			sum += i;
		}
		return sum;
	}

	public boolean push(int stackNum, int value) {
		// if array is full
		if (getTotalSize() == stackArr.length) {
			return false;
		}

		// target stack is empty
		if (stackTop[stackNum] == -1) {
			stackTop[stackNum] = 0;
			for (int i = stackNum - 1; i >= 0; i--) {
				if (stackTop[i] != -1) {
					stackTop[stackNum] = stackTop[i] + 1;
					break;
				}
			}
			stackTop[stackNum] = stackTop[stackNum];
			// shift right
			for (int i = getTotalSize(); i > stackTop[stackNum]; i--) {
				stackArr[i] = stackArr[i - 1];
			}
			stackArr[stackTop[stackNum]] = value;
			for (int i = stackNum + 1; i < stackTop.length; i++) {
				if (stackTop[i] != -1) {
					stackTop[i] += 1;
				}
			}
		} else {
			for (int i = stackNum; i < stackTop.length; i++) {
				if (stackTop[i] != -1) {
					stackTop[i] += 1;
				}
			}
			for (int i = getTotalSize(); i > stackTop[stackNum]; i--) {
				stackArr[i] = stackArr[i - 1];
			}
			stackArr[stackTop[stackNum]] = value;
		}
		size[stackNum]++;
		return true;
	}

	public int pop(int stackNum) throws EmptyStackException {
		// use if-else control flow
		// if (stackTop[stackNum] == -1) {
		// throw new EmptyStackException();
		// } else {
		// int value = stackArr[stackTop[stackNum]];
		// if (getStackSize(stackNum) == 1) {
		// stackTop[stackNum] = -1;
		// for (int i = stackNum + 1; i < stackTop.length; i++) {
		// if (stackTop[i] != -1) {
		// stackTop[i]--;
		// }
		// }
		// } else {
		// for (int i = stackNum; i < stackTop.length; i++) {
		// if (stackTop[i] != -1) {
		// stackTop[i]--;
		// }
		// }
		// }
		// return value;
		// }
		// use try-catch control flow (* first time know how to use it)
		try {
			// exception may be thrown here
			int value = stackArr[stackTop[stackNum]];

			// shift left
			for (int i = stackTop[stackNum]; i < getTotalSize() - 1; i++) {
				stackArr[i] = stackArr[i + 1];
			}

			// update stack head pointers
			if (getStackSize(stackNum) == 1) {
				stackTop[stackNum] = -1;
			} else {
				stackTop[stackNum]--;
			}
			for (int i = stackNum + 1; i < stackTop.length; i++) {
				if (stackTop[i] != -1) {
					stackTop[i]--;
				}
			}

			size[stackNum]--;
			return value;
		} catch (ArrayIndexOutOfBoundsException e) {
			throw new EmptyStackException();
		}
	}

	public int peek(int stackNum) throws EmptyStackException {
		// if-else
		// if (stackTop[stackNum] == -1) {
		// throw new EmptyStackException();
		// } else {
		// return stackArr[stackTop[stackNum]];
		// }
		// try-catch
		try {
			return stackArr[stackTop[stackNum]];
		} catch (ArrayIndexOutOfBoundsException e) {
			throw new EmptyStackException();
		}
	}

	public boolean isEmpty(int stackNum) {
		return stackTop[stackNum] == -1;
	}

	public void printStacks() {
		System.out.println("Printing stacks...");
		// print stack array
		for (int i = 0; i < getTotalSize(); i++) {
			System.out.print(stackArr[i] + " ");
		}
		System.out.print("| ");
		for (int i = getTotalSize(); i < stackArr.length; i++) {
			System.out.print(stackArr[i] + " ");
		}
		System.out.println();

		// print stacks head pointer
		for (int i = 0; i < stackTop.length; i++) {
			System.out.print(stackTop[i] + " ");
		}
		System.out.println();

		// print stacks
		int current = 0;
		for (int i = 0; i < stackTop.length; i++) {
			System.out.print("stack " + i + ": ");
			if (stackTop[i] != -1) {
				for (int j = current; j <= stackTop[i]; j++) {
					System.out.print(stackArr[j] + " ");
				}
				current = stackTop[i] + 1;
			}
			System.out.println();
		}
		System.out.println();
	}

	public static void main(String[] args) {
		SingleArrayStack sd = new SingleArrayStack(100, 2);

		sd.push(0, 1);
		sd.printStacks();
		sd.push(0, 2);
		sd.printStacks();
		sd.push(0, 3);
		sd.printStacks();
		System.out.println("peek(0): " + sd.peek(0) + "\n");

		sd.push(1, 1);
		sd.printStacks();
		sd.push(1, 2);
		sd.printStacks();
		sd.push(1, 3);
		sd.printStacks();
		System.out.println("peek(1): " + sd.peek(0) + "\n");

		sd.push(0, 4);
		sd.printStacks();
		sd.push(0, 5);
		sd.printStacks();
		sd.push(0, 6);
		sd.printStacks();
		System.out.println("peek(0): " + sd.peek(0) + "\n");

		sd.pop(0);
		sd.pop(1);
		System.out.println("pop(0) + poo(1)");
		sd.printStacks();
	}
}
