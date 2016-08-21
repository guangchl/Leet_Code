package chapter_3_StacksAndQueues;

import java.util.ArrayList;
import java.util.Stack;

/** 3.4 Tower of Hanoi */
public class HanoiTower {
	ArrayList<Stack<Integer>> towers;

	public HanoiTower(int n) {
		towers = new ArrayList<Stack<Integer>>();
		Stack<Integer> s1 = new Stack<Integer>();
		Stack<Integer> s2 = new Stack<Integer>();
		Stack<Integer> s3 = new Stack<Integer>();
		for (int i = n; i > 0; i--) {
			s1.push(i);
		}
		towers.add(s1);
		towers.add(s2);
		towers.add(s3);
	}

	public void hanoiMove(int fromTower, int toTower, int tempTower, int n) {
		if (n == 1) {
			towers.get(toTower).push(towers.get(fromTower).pop());
			return;
		} else {
			hanoiMove(fromTower, tempTower, toTower, n - 1);
			hanoiMove(fromTower, toTower, tempTower, 1);
			hanoiMove(tempTower, toTower, fromTower, n - 1);
		}
	}
	
	public void printTowers() {
		System.out.println("Printing...");
		int j = 1;
		for (Stack<Integer> tower : towers) {
			System.out.print("Tower " + (j++) + ": ");
			for (int i : tower) {
				System.out.print(i + " ");
			}
			System.out.println();
		}
	}

	public static void main(String[] args) {
		int n = 30;
		HanoiTower ht = new HanoiTower(n);
		ht.printTowers();
		
		long startTime = System.currentTimeMillis();
		ht.hanoiMove(0, 2, 1, n);
		long endTime = System.currentTimeMillis();
		
		ht.printTowers();
		System.out.println("Time cost in milliseconds: " + (endTime-startTime));

	}

}
