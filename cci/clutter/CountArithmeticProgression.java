package clutter;

public class CountArithmeticProgression {

	public int count(int[] A) {
		if (A == null || A.length < 3) {
			return 0;
		}
		
		int result = 0;
		int cnt = 2;
		int diff = A[1] - A[0];
		
		for (int i = 2; i < A.length; i++) {
			if (A[i] - A[i - 1] == diff) {
				cnt++;
			} else {
				result += (cnt - 2) * (cnt - 1) / 2;
				
				// start next round, update variable
				cnt = 2;
				diff = A[i] - A[i - 1];
			}
		}
		
		if (cnt > 2) {
			result += (cnt - 2) * (cnt - 1) / 2;
		}
		
		return result;
	}

	public void test() {
		int[] A = { 1, 1, 1, 2, 4, 6, 8 };
		System.out.println(count(A));
	}
	
	public static void main(String[] args) {
		CountArithmeticProgression cap = new CountArithmeticProgression();
		cap.test();
	}
}
