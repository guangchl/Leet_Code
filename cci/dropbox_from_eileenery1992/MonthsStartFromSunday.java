package dropbox_from_eileenery1992;

/**
 * Given some certain day in the year is one from Sunday to Monday in the week,
 * calculate the number of months which start with Sunday from fromYear to
 * toYear.
 */
public class MonthsStartFromSunday {
	private static int YEAR_YYYY = 2013, MONTH_MM = 11, DAY_DD = 16;
	private static int REMAINDER = 6;
	private static final int[] REMAINDERS_MONTH = { 3, 0, 3, 2, 3, 2, 3,
			3, 2, 3, 2, 3 };
	private static final int[] REMAINDERS_MONTH_LEAPYEAR = { 3, 1, 3, 2, 3, 2,
			3, 3, 2, 3, 2, 3 };
	private static final int REMAINDER_YEAR = 1;
	private static final int REMAINDER_LEAPYEAR = 1;

	// return if the year is 
	private static boolean isLeapYear(int year) {
		return (year % 400 == 0) || (year % 100 != 0 && year % 4 == 0);
	}
	
	public static void printResult(int fromYear, int toYear, int result) {
		System.out.println("From " + fromYear + " to " + toYear
				+ ", there are " + result + " months start from Sunday.\n");
	}

	public static int count(int fromYear, int toYear) {
		// find what is the 1st day
		REMAINDER = REMAINDER - DAY_DD + 1;
		DAY_DD = 1;
		// find what is the 1st day of the 1st month
		// deduct last month remainder at each time
		if (isLeapYear(fromYear)) {
			for (int i = MONTH_MM - 1; i > 0; i--) {
				REMAINDER = REMAINDER - REMAINDERS_MONTH_LEAPYEAR[i - 1];
			}
		} else {
			for (int i = MONTH_MM - 1; i > 0; i--) {
				REMAINDER = REMAINDER - REMAINDERS_MONTH[i - 1];
			}
		}
		MONTH_MM = 1;
		// find what is the 1st day of the 1st month of the fromYear
		for (int i = YEAR_YYYY - 1; i >= fromYear; i--) {
			if (isLeapYear(i)) {
				REMAINDER -= REMAINDER_LEAPYEAR;
			} else {
				REMAINDER -= REMAINDER_YEAR;
			}
		}
		YEAR_YYYY = fromYear;

		// Normalize the REMAINDER
		REMAINDER = REMAINDER % 7;
		if (REMAINDER < 0)
			REMAINDER += 7;

		// calculate from fromYear to toYear
		int counter = 0;
		int rmd = REMAINDER;
		int[] remainders;
		for (int i = fromYear; i <= toYear; i++) {
			if (isLeapYear(i)) {
				remainders = REMAINDERS_MONTH_LEAPYEAR;
			} else {
				remainders = REMAINDERS_MONTH;
			}
			for (int remainder : remainders) {
				if (rmd % 7 == 0) counter++;
				rmd += remainder;
			}
		}
		return counter;
	}

	public static void main(String[] args) {
		int fromYear, toYear, result;
		fromYear = 2013;
		toYear = 2013;
		result = MonthsStartFromSunday.count(fromYear, toYear);
		printResult(fromYear, toYear, result);
	}

}
