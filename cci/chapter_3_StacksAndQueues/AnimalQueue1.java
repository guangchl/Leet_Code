package chapter_3_StacksAndQueues;

import java.util.LinkedList;
import java.util.NoSuchElementException;

/** 3.7 Animal Shelter which can enqueue, dequeueAny, dequeueCat, dequeueDog */
public class AnimalQueue1 implements AnimalQueue {
	private LinkedList<Animal> queue;
	
	public AnimalQueue1() {
		queue = new LinkedList<Animal>();
	}
	
	public void enqueue(Animal a) {
		queue.add(a);
	}
	
	public Animal dequeueAny() {
		if (queue.size() > 0) {
			return queue.remove(0);
		} else {
			throw new NoSuchElementException();
		}
		
	}
	
	public Dog dequeueDog() {
		int i = 0;
		while (i < queue.size() && !(queue.get(i) instanceof Dog)) {
			i++;
		}
		
		if (i < queue.size()) {
			return (Dog)queue.remove(i);
		} else {
			throw new NoSuchElementException();
		}
	}
	
	public Cat deueueCat() {
		int i = 0;
		while (i < queue.size() && !(queue.get(i) instanceof Cat)) {
			i++;
		}
		
		if (i < queue.size()) {
			return (Cat)queue.remove(i);
		} else {
			throw new NoSuchElementException();
		}
	}
	
	public void printQueue() {
		for (Animal animal : queue) {
			System.out.print(animal.getType() + " ");
		}
		System.out.println();
	}
	
}
