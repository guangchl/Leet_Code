package design;

/**
 * In a Restaurant, there are tables with different seating arrangement.
 * Arrangement are as follows: Tables which can seat 2 people - T1 Tables which
 * can seat 4 people - T2 Tables which can seat 8 people - T3 In case there is a
 * group which contains more than 8 people, they can be seated by rearranging T1
 * and T2 (T3 cannot be rearranged). Also, there is a finite number of T1, T2
 * and T3.
 * 
 * Groups of people are coming in (as in a stream). Task is to seat them based
 * on the number of people in a group. Write an algorithm to fulfill this task
 * and wait time should be minimized.
 * 
 * Example : Group1 contains 2 people. Algorithm assigns them a T1. Group2
 * contains 3 people. Algorithm assigns them a T2. Group3 contains 4 people.
 * Algorithm assigns them a T2. Group4 contains 10 people. Algorithm assigns
 * them a rearrangement of 2 * T2 + 1 * T1 or 5 * T1 or 3 * T1 + 1 * T2
 * 
 * Parallel task processing is encouraged.
 */
public class TableManager {
	private static int T1 = 15;
	private static int T2 = 10;
	private static int T3 = 2;
	//private int left1 = 0;
	//private int left2 = 0;
	//private int left3 = 0;

	private static final int SIZE1 = 2;
	private static final int SIZE2 = 4;
	private static final int SIZE3 = 8;

	Table[] tables = new Table[T1 + T2 + T3];
	boolean[] inUse = new boolean[T1 + T2 + T3];
	
	public void build() {
		TableFinder tf = new TableFinder(this);
		TableCleaner tc = new TableCleaner(this);

		Thread thread1 = new Thread(tf);
		Thread thread2 = new Thread(tc);
		
		thread1.start();
		thread2.start();
		
		int i;
		for (i = 0; i < T1; i++) {
			tables[i] = new Table(SIZE1);
		}
		for (; i < T1 + T2; i++) {
			tables[i] = new Table(SIZE2);
		}
		for (; i < T1 + T2 + T3; i++) {
			tables[i] = new Table(SIZE3);
		}
	}

	public boolean findPlace(int target) {

		return false;
	}

	public class TableFinder implements Runnable {
		private TableManager tm;
		public int targetNumber;

		public void findTable(int target) throws InterruptedException {
			synchronized (tm) {
				while (tm.findPlace(target) == false)
					tm.wait();
			}
		}

		public TableFinder(TableManager tm) {
			this.tm = tm;
		}

		@Override
		public void run() {
			try {
				findTable(targetNumber);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	public class TableCleaner implements Runnable {
		private TableManager tm;

		public void cleanTable() throws InterruptedException {
            synchronized (tm) {
            	int i = 0;
            	for (Table table : tm.tables) {
            		if (table.done()) {
            			inUse[i++] = false;
            		}
            	}
            }
        }

		public TableCleaner(TableManager tm) {
			this.tm = tm;
		}

		@Override
		public void run() {
			try {
				cleanTable();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	public class Table {
		public int size;
		public int millis = 0;
		
		public Table(int size) {
			this.size = size;
		}
		
		public boolean done() {
			// do something here
			
			return false;
		}
	}
	
	public static void main(String[] args) {
		TableManager tm = new TableManager();
		tm.build();
	}

}
