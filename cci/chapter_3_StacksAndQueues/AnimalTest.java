package chapter_3_StacksAndQueues;

public class AnimalTest {
	
	public static void doTest(AnimalQueue aq) {
		System.out.println("Testing " + aq.getClass().getName() + "...");
		aq.enqueue(new Cat());
		aq.enqueue(new Cat());
		aq.enqueue(new Dog());
		aq.enqueue(new Cat());
		aq.enqueue(new Dog());
		aq.enqueue(new Dog());
		aq.enqueue(new Cat());
		aq.enqueue(new Cat());
		aq.enqueue(new Dog());
		aq.printQueue();
		
		System.out.println("dequeueAny()...");
		aq.dequeueAny();
		aq.printQueue();
		
		System.out.println("dequeueDog()...");
		aq.dequeueDog();
		aq.printQueue();
		
		System.out.println("dequeueDog()...");
		aq.dequeueDog();
		aq.printQueue();
		
		System.out.println();
	}

	public static void main(String[] args) {
		AnimalQueue aq1 = new AnimalQueue1();
		AnimalQueue aq2 = new AnimalQueue2();
		doTest(aq1);
		doTest(aq2);
	}

}
