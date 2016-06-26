package clutter;

import java.util.StringTokenizer;

/**
 * Solution in Python
 * sentence = "My name is X Y Z"
 * words = ip.split()
 * words.reverse()
 * print ' '.join(words)
 * 
 */
public class ReverseWordsInString {
	
	public String reverse(String sentence) {
		StringBuilder result = new StringBuilder();
		
	    StringTokenizer tokens = new StringTokenizer(sentence, " ");
	    while (tokens.hasMoreTokens()) {
	        StringBuilder word = new StringBuilder(tokens.nextToken());
	        result.append(word.reverse() + " ");
	    }
	    
	    return result.substring(0, result.length() - 1);
	}
	
	public String reverse2(String str) {
		char charArray[] = str.toCharArray();
		for (int i = 0; i < str.length(); i++) {
			if (charArray[i] == ' ')
				return reverse2(str.substring(i + 1)) + str.substring(0, i)
						+ " ";
		}

	    return str;
	}
	
	public void test() {
		String s = "I am Guangcheng Lu";
		System.out.println(reverse(s));
	}

	public static void main(String[] args) {
		ReverseWordsInString solution = new ReverseWordsInString();
		solution.test();

	}

}
