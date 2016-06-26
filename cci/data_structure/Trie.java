package data_structure;

import java.util.*;

/**
 * Tries���ݽṹ���ֵ����� ����ֻ���������ʵ�֣����ж�ĳ�������Ƿ���ֹ������ʳ���Ƶ���ȣ�������Ҫ����������չ
 * 
 * @author
 * 
 */
public class Trie {

	private static TrieNode root;
	private char[] characterTable = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
			'i', 'j', 'k',
			'l', // ������ʱ��ʹ��
			'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
			'z' };

	public Trie() {
		root = new TrieNode();
	}

	public static void main(String[] args) {

		Trie trie = new Trie();
		trie.insert("china");
		trie.insert("cinema");
		boolean exist = trie.find("chi");
		if (exist)
			System.out.println("there is a prefix");
		trie.traverse();
	}

	/**
	 * �����ַ���
	 * 
	 * @param word
	 */
	public void insert(String word) {
		TrieNode node = root;
		word = word.trim();
		for (int i = 0; i < word.length(); i++) {
			if (!(node.children.containsKey(word.charAt(i)))) {
				node.children.put(word.charAt(i), new TrieNode());
			}
			node = node.children.get(word.charAt(i));
		}
		node.terminable = true;
		node.count++;
	}

	/**
	 * ����ĳ���ַ���
	 * 
	 * @param word
	 * @return
	 */
	public boolean find(String word) {
		TrieNode node = root;
		for (int i = 0; i < word.length(); i++) {
			if (!(node.children.containsKey(word.charAt(i)))) {
				return false;
			} else {
				node = node.children.get(word.charAt(i));
			}
		}
		return true;
		// return node.terminable; // ������ַ�����Trie·���У�Ҳ����˵���õ����Ѵ��ڣ���Ϊ���п�����ĳ���Ӵ�
	}

	/**
	 * ɾ��ĳ���ַ����������Ǹ����ʣ�������ǰ׺��
	 * 
	 * @param word
	 */
	public void delete(String word) {
		if (!find(word)) {
			System.out.println("no this word.");
			return;
		}
		TrieNode node = root;
		deleteStr(node, word);
	}

	public boolean deleteStr(TrieNode node, String word) {
		if (word.length() == 0) {
			node.terminable = false; // ��Ҫ����������Ϣ�ĸ���
			return node.children.isEmpty();
		}
		if (deleteStr(node.children.get(word.charAt(0)), word.substring(1))) {
			node.children.remove(word.charAt(0));
			if (node.children.isEmpty() && node.terminable == false) { // ע��ڶ���������������"abcd"��"abc",ɾ��abcdʱ��Ҫ�ж��м�·�����ǲ�����һ���Ӵ��Ľ���
				return true;
			}
		}
		return false;
	}

	/**
	 * ���ֵ������Tire�����г��ֵĵ��ʼ�Ƶ��
	 */
	public void traverse() {
		StringBuffer word = new StringBuffer("");
		TrieNode node = root;
		traverseTrie(node, word);
	}

	public void traverseTrie(TrieNode node, StringBuffer word) {
		if (node.terminable) {
			System.out.println(word + "------" + node.count);
			if (node.children.isEmpty())
				return;
		}
		for (int i = 0; i < characterTable.length; i++) {
			if (!(node.children.containsKey(characterTable[i])))
				continue;
			traverseTrie(node.children.get(characterTable[i]),
					word.append(characterTable[i]));
			word.deleteCharAt(word.length() - 1);
		}
	}
}

/**
 * Trie�����
 * 
 * @author
 * @param <T>
 */
class TrieNode {
	public boolean terminable; // �ǲ��ǵ��ʽ�β����Ҷ�ӽڵ㣩
	public int count; // ���ʳ���Ƶ��
	public Map<Character, TrieNode> children = null;

	public TrieNode() {
		terminable = false;
		count = 0;
		children = new HashMap<Character, TrieNode>();
	}
}
