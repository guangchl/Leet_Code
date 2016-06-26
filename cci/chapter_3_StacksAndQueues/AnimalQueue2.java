package chapter_3_StacksAndQueues;

import java.util.LinkedList;
import java.util.NoSuchElementException;

/** 3.7 Animal Shelter which can enqueue, dequeueAny, dequeueCat, dequeueDog */
public class AnimalQueue2 implements AnimalQueue {
	private LinkedList<Animal> catQueue;
	private LinkedList<Animal> dogQueue;
	private int order;

	public AnimalQueue2() {
		catQueue = new LinkedList<Animal>();
		dogQueue = new LinkedList<Animal>();
		order = 0;
	}

	public void enqueue(Animal a) {
		if (a instanceof Cat) {
			a.setOrder(order++);
			catQueue.add(a);
		} else if (a instanceof Dog) {
			a.setOrder(order++);
			dogQueue.add(a);
		} else {
			throw new IllegalArgumentException();
		}
	}

	public Animal dequeueAny() {
		if (catQueue.size() > 0) {
			return catQueue.peek().isOlder(dogQueue.peek()) ? catQueue.remove()
					: dogQueue.remove();
		} else if (dogQueue.size() > 0) {
			return dogQueue.remove();
		} else {
			throw new NoSuchElementException();
		}
	}

	public Dog dequeueDog() {
		if (dogQueue.size() > 0) {
			return (Dog) dogQueue.remove();
		} else {
			throw new NoSuchElementException();
		}
	}

	public Cat deueueCat() {
		if (catQueue.size() > 0) {
			return (Cat) catQueue.remove();
		} else {
			throw new NoSuchElementException();
		}
	}

	public void printSingleQueue(LinkedList<Animal> queue, int from) {
		for (int i = from; i < queue.size(); i++) {
			System.out.print(queue.get(i).getType() + " ");
		}
	}

	public void printQueue() {
		int i1 = 0;
		int i2 = 0;
		while (i1 < catQueue.size() || i2 < dogQueue.size()) {
			if (i1 < catQueue.size() && i2 >= dogQueue.size()) {
				printSingleQueue(catQueue, i1);
				break;
			} else if (i1 >= catQueue.size() && i2 < dogQueue.size()) {
				printSingleQueue(dogQueue, i2);
				break;
			} else {
				Animal a = catQueue.get(i1).isOlder(dogQueue.get(i2)) ? catQueue
						.get(i1++) : dogQueue.get(i2++);
				System.out.print(a.getType() + " ");
			}
		}
		System.out.println();
	}

}
