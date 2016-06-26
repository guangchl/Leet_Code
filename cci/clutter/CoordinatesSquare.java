package clutter;

import java.util.ArrayList;
import java.util.List;

public class CoordinatesSquare {
	/** Definition of a Point */
	private class Point {
		int x;
		int y;
		
		public Point(int x, int y) {
			this.x = x;
			this.y = y;
		}
	}
	
	public boolean isSquare(List<Point> listPoints) {
		if (listPoints != null && listPoints.size() == 4) {
			int distance1 = distance(listPoints.get(0), listPoints.get(1));
			int distance2 = distance(listPoints.get(0), listPoints.get(2));
			int distance3 = distance(listPoints.get(0), listPoints.get(3));

			if (distance1 == distance2) {
				// checking if the sides are equal to the diagonal
				if (distance3 == distance1 + distance2) {
					return true;
				}

			} else if (distance1 == distance3) {
				// checking if the sides are equal to the diagonal
				if (distance2 == distance1 + distance3) {
					return true;
				}

			}
		}
		return false;
	}

	private static int distance(Point point, Point point2) {
		// (x2-x1)^2+(y2-y1)^2
		return (int) (Math.pow(point2.x - point.x, 2) + (Math.pow(point2.y
				- point.y, 2)));
	}

	public void test() {
		List<Point> pointz = new ArrayList<Point>();
		pointz.add(new Point(2, 2));
		pointz.add(new Point(2, 4));
		pointz.add(new Point(4, 2));
		pointz.add(new Point(4, 4));
		System.out.println(isSquare(pointz));
	}
	public static void main(String[] args) {
		CoordinatesSquare sol = new CoordinatesSquare();
	    sol.test();
	}

}
