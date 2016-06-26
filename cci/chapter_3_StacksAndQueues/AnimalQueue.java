package chapter_3_StacksAndQueues;

public interface AnimalQueue {

	public void enqueue(Animal a);
	
	public Animal dequeueAny();
	
	public Dog dequeueDog();
	
	public Cat deueueCat();
	
	public void printQueue();
		
}
