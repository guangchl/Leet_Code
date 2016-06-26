package pastinterviews;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

/**
 * Problem from DynamicSignals & Aspera - Find Longest Word Made of Other Words
 * 
 * Write a program that reads a file containing a sorted list of words (one word
 * per line, no spaces, all lower case), then identifies the 
 * 1. 1st longest word in the file that can be constructed by concatenating 
 * copies of shorter words also found in the file.
 * 2. The program should then go on to report the 2nd longest word found
 * 3. Total count of how many of the words in the list can be constructed of 
 * other words in the list.
 * 
 * Author: Guangcheng Lu
 * All rights reserved.
 */
public class WordCombinations {
	/** class that encapsulates all required output */
	public class Result {
		private String[] words;
		private int count;
		
		public Result(String[] words, int count) {
			this.words = words;
			this.count = count;
		}

		/** @return the words */
		public String[] getWordArray() {
			return words;
		}
		
		/** @return the count */
		public int getCount() {
			return count;
		}
	}

	/**
	 * The primary method for this problem
	 * 
	 * Instruction:
	 *  1) Construct a new project and copy this file to a package
	 *  2) Change the package name, or delete package name for default package
	 *  3) Put the file to process under the root directory of the Java project.
	 *  4) Change the file name in test() to be the file you want to process.
	 * 
	 * @return Result Object that store the 3 output
	 */
	private static int minLength;
	public Result findCombinations(String fileName, int longestN) {
		String[] words = new String[longestN];
		int count = 0;
		Map<String, Boolean> dictionary = new HashMap<String, Boolean>();
		List<String> wordList = new LinkedList<String>();
		
		// read the file and store the words in the dictionary and word list
		try {
			Scanner in = new Scanner(new FileReader(fileName));
			while (in.hasNext()) {
				String s = in.next();
				dictionary.put(s, true);
				wordList.add(s);
			}
			in.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.out.println("Error: FileNotFound. Please check the path of "
					+ "the file and the file name, then run the program again");
			return null;
		}

		// sort the word list by length in descending order
		Collections.sort(wordList, new Comparator<String>() {
			@Override
			public int compare(String o1, String o2) {
				if (o1.length() < o2.length()) {
					return 1;
				} else if (o1.length() > o2.length()) {
					return -1;
				}
				return 0;
			}
		});
		
		// find the MIN_LENGTH
		minLength = wordList.get(wordList.size() - 1).length();
		
		// traverse the list to find the result
		int position = 0;
		for (String word : wordList) {
			if (canBeBuilt(word, dictionary, false)) {
				if (position < longestN) {
					words[position] = word;
					position++;
				}
				count++;
			}
		}
		
		return new Result(words, count);
	}
	
	private boolean canBeBuilt(String word, Map<String, Boolean> dictionary,
			boolean isSubString) {
		if (dictionary.containsKey(word) && isSubString) {
			return dictionary.get(word);
		}
		
		for (int i = minLength; i < word.length() - minLength; i++) {
			String left = word.substring(0, i);
			String right = word.substring(i);
			if (dictionary.containsKey(left) && dictionary.get(left) &&
					canBeBuilt(right, dictionary, true)) {
				return true;
				//dictionary.put(word, true); // dynamic programming
			}
		}
		
		//dictionary.put(word, false); // dynamic programming
		return false;
	}
	
	/** Test use the given document, and print the results */
	public void test() {
		long start = System.currentTimeMillis(); // Timer starts
		
		// search the words
		String fileName = "src/pastInterviews/wordsforproblem.txt";
		int longestN = 2;
		Result result = findCombinations(fileName, longestN);
		
		// print result
		System.out.println("Longest " + longestN + " words:");
		String[] words = result.getWordArray();
		for (String word : words) {
			System.out.println("  " + word);
		}
		System.out.println("\nTotal count: " + result.getCount());
		
		System.out.println("\nTime elapse: " + (System.currentTimeMillis() - start) + " milliseconds"); // Timer ends
	}

	public static void main(String[] args) {
		WordCombinations s = new WordCombinations();
		s.test();
	}
}