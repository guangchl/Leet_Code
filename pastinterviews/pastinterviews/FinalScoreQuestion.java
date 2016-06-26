package pastinterviews;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class FinalScoreQuestion {
	class TestResult {
		int studentId;
		String testDate = null;
		int testScore;

		TestResult(int studentId, int testScore) {
			this.studentId = studentId;
			this.testScore = testScore;
		}
	}

	class PQAndAverage {
		PriorityQueue<Integer> pq = new PriorityQueue<>();
		Double avg = 0.0;
		static final int maxTestToConsider = 5;

		public void add(int i) {
			if (pq.isEmpty()) {
				pq.add(i);
				avg = (double) i;
			} else if (pq.size() < maxTestToConsider) {
				avg = (((avg * pq.size()) + i) / (pq.size() + 1));
				pq.add(i);
			} else {
				if (pq.peek() < i) {
					int temp = pq.peek();
					avg = (((avg * pq.size()) + i - temp) / (pq.size()));
					pq.poll();
					pq.add(i);
				}
			}
		}

		@Override
		public String toString() {
			return ("AVG :" + avg + ", Queue:" + pq);
		}
	}

	Map<Integer, Double> calculateFinalScores(List<TestResult> results) {
		Map<Integer, Double> result = new HashMap<Integer, Double>();
		Map<Integer, PQAndAverage> avg = new HashMap<Integer, PQAndAverage>();

		for (TestResult testResult : results) {
			if (!avg.containsKey(testResult.studentId)) {
				PQAndAverage tempPqWithAverage = new PQAndAverage();
				tempPqWithAverage.add(testResult.testScore);
				avg.put(testResult.studentId, tempPqWithAverage);
			} else {
				PQAndAverage temp = avg.get(testResult.studentId);
				temp.add(testResult.testScore);
			}
		}

		Iterator<Integer> iterator;
		for (iterator = avg.keySet().iterator(); iterator.hasNext();) {
			int studentId = iterator.next();
			result.put(studentId, avg.get(studentId).avg);
		}

		return result;
	}
}
