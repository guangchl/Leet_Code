package chapter_9_;

import java.util.ArrayList;
import java.util.Arrays;

public class Main {

	public class Box implements Comparable<Box>{
		int height;
		int width;
		int length;
		
		public Box(int height, int width, int length) {
			this.height = height;
			this.width = width;
			this.length = length;
		}

		@Override
		public int compareTo(Box b) {
			if (length < b.length) return 1;
			if (length == b.length) return 0;
			return -1;
		}
	}
	
	/** store the result as the indexes of the boxes */
	public class BoxStack {
		ArrayList<Integer> stack;
		int h;
		
		public BoxStack(int h, ArrayList<Integer> stack) {
			this.stack = stack;
			this.h = h;
		}
	}
	
	public ArrayList<Box> createStack(Box[] boxes) {
		ArrayList<Box> tower = new ArrayList<Box>();
		int len = boxes.length;
		if (len == 0) return tower;
		Arrays.sort(boxes);
		
		// dp[i] store result from 0 to i
		BoxStack[] dp = new BoxStack[len];
		BoxStack firstStack = new BoxStack(boxes[0].height, new ArrayList<Integer>());
		firstStack.stack.add(0);
		dp[0] = firstStack;
		
		// calculate every dp[i]
		for (int i = 1; i < len; i++) {
			int maxIndex = -1;
			int maxHeight = 0;
			for (int j = i - 1; j >= 0; j++) {
				if (boxes[i].width > boxes[j].width) {
					int newHeight = dp[j].h + boxes[i].height;
					if (newHeight > maxHeight) {
						maxHeight = newHeight;
						maxIndex = j;
					}
				}
			}
			if (maxIndex != -1) {
				BoxStack bs = new BoxStack(maxHeight, new ArrayList<Integer>(dp[maxIndex].stack));
				bs.stack.add(i);
				dp[i] = bs;
			}
		}
		
		// add the box instances by their index
		for (Integer i : dp[len - 1].stack) {
			tower.add(boxes[i]);
		}
		
		return tower;
	}
}
