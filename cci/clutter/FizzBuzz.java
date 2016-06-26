package clutter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;

public class FizzBuzz {

	public void fizzBuzz(int start, int end) {
		if (start > end ) {
			return;
		}
		
        for (int i = start; i <= end; i++) {
            if (i % 15 == 0) {
            	System.out.println("FizzBuzz");
            } else if (i % 3 == 0) {
            	System.out.println("Fizz");
            } else if (i % 5 == 0) {
            	System.out.println("Buzz");
            } else {
            	System.out.println(i);
            }
        }
    }
	
	private class FB implements Comparable<FB> {
		private int base;
		private int current;
		private String string;
		
		public FB(int base, String string) {
			this.base = base;
			current = base;
			this.string = string;
		}
		
		public int getCurrent() {
			return current;
		}
		
		public String getString() {
			return string;
		}
		public void updateCurrent() {
			current += base;
		}

		@Override
		public int compareTo(FB o) {
			if (current > o.current) {
				return 1;
			} else if (current < o.current) {
				return -1;
			} else {
				if (base > o.base) {
					return 1;
				} else if (base < o.base) {
					return -1;
				} else {
					return 0;
				}
			}
		}
	}
	
	public void fizzBuzz2(int start, int end, Map<Integer, String> dict) {
		// keep a queue store all current multiple of integers
		PriorityQueue<FB> pq = new PriorityQueue<FB>();
		for (int i : dict.keySet()) {
			i = ((start - 1) / i + 1) * i;
			pq.offer(new FB(i, dict.get(i)));
		}
		
		int current = pq.peek().getCurrent();
		while (current <= end) {
			// store list of all FB object for current number
			ArrayList<FB> fbList = new ArrayList<FB>();
			while (pq.size() > 0 && pq.peek().getCurrent() == current) {
				fbList.add(pq.poll());
			}
			
			// print result
			System.out.print(current + ": ");
			for (FB fb : fbList) {
				System.out.print(fb.getString() + " ");
			}
			System.out.println();
			
			// update pq and current
			for (FB fb : fbList) {
				fb.updateCurrent();
				pq.offer(fb);
			}
			current = pq.peek().getCurrent();
		}
	}
	
	public void test() {
		Map<Integer, String> dict = new HashMap<Integer, String>();
		dict.put(3, "Fizz");
		dict.put(5, "Buzz");
		
		fizzBuzz2(1, 100, dict);
	}
	
	public static void main(String[] args) {
		FizzBuzz fizzBuzz = new FizzBuzz();
		fizzBuzz.test();
	}

}
