package chapter_3_StacksAndQueues;

public abstract class Animal {
	protected String type;
	protected int order; // used for AnimalQueue2
	
	public String getType() {
		return type;
	}
	
	// ********************** USED FOR ANIMALQUEUE2 **********************
	public void setOrder(int order) {
		this.order = order;
	}
	
	public int getOrder() {
		return order;
	}
	
	public boolean isOlder(Animal a) {
		if (a != null) {
			return order < a.order;
		} else {
			return true;
		}
		
	}
}
