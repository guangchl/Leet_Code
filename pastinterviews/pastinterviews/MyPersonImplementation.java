package pastinterviews;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;

/**
 * Interviw from Rational Systems
 * Guangcheng Lu
 * Andrew ID: guangchl
 *
 */
public class MyPersonImplementation {
	public interface Person {

		static final char MALE = 'M';
		static final char FEMALE = 'F';

		// name
		String getName();

		void setName(String name);

		// ssn
		long getSSN();

		void setSSN(long ssn);

		// gender
		char getGender();

		void setGender(char gender);

		// relationships
		void addParent(Person parent);

		Set<Person> getParents();

		void addChild(Person child);

		Set<Person> getChildren();

		void setSpouse(Person spouse);

		Person getSpouse();

		// returns true if the person is married
		boolean hasSpouse();

		// returns true if the person is related, false otherwise
		boolean isRelated(Person person);

	}

	public class MyPerson implements Person {

		private String name;
		private long ssn;
		private char gender;

		// I add the following necessary fields
		private Set<Person> parents;
		private Set<Person> children;
		private Person spouse;

		public String getName() {
			return name;
		}

		public void setName(String name) {
			this.name = name;
		}

		public long getSSN() {
			return ssn;
		}

		public void setSSN(long ssn) {
			this.ssn = ssn;
		}

		public char getGender() {
			return gender;
		}

		public void setGender(char gender) {
			this.gender = gender;
		}

		/**
		 * Constructor
		 * 
		 * @param name
		 * @param ssn
		 * @param gender
		 */
		public MyPerson(String name, long ssn, char gender) {
			this.name = name;
			this.ssn = ssn;
			this.gender = gender;

			// initialize the addition fields
			parents = new HashSet<Person>();
			children = new HashSet<Person>();
		}

		// ****************** I have filled the following methods
		// ******************

		public void addParent(Person parent) {
			parents.add(parent);
			parent.getChildren().add(this);
		}

		public Set<Person> getParents() {
			return parents;
		}

		public void addChild(Person child) {
			children.add(child);
			child.getParents().add(this);
		}

		public Set<Person> getChildren() {
			return children;
		}

		public void setSpouse(Person spouse) {
			this.spouse = spouse;

			// to avoid the infinite loop
			if (!spouse.hasSpouse() || spouse.getSpouse().getSSN() != this.ssn) {
				spouse.setSpouse(this);
			}
		}

		public Person getSpouse() {
			return spouse;
		}

		public boolean hasSpouse() {
			return spouse != null;
		}

		/**
		 * Do breath first search using a queue.
		 * 
		 * Here I assume the same person is also a relationship.
		 */
		public boolean isRelated(Person person) {
			// construct a queue which store the persons that are pending to
			// check
			Queue<Person> queue = new LinkedList<Person>();
			queue.offer(this);

			// construct a set to store all ssn that we have checked
			Set<Long> ssnSet = new HashSet<Long>();

			while (!queue.isEmpty()) {
				// first person in the queue
				Person p = queue.poll();

				// relationship found
				if (p.getSSN() == person.getSSN()) {
					return true;
				}

				// update ssn set
				ssnSet.add(p.getSSN());

				// add parents to the queue
				for (Person parent : p.getParents()) {
					if (!ssnSet.contains(parent.getSSN())) {
						queue.offer(parent);
					}
				}

				// add children to the queue
				for (Person child : p.getChildren()) {
					if (!ssnSet.contains(child.getSSN())) {
						queue.offer(child);
					}
				}

				// add spouse to the queue
				if (p.hasSpouse() && !ssnSet.contains(p.getSpouse().getSSN())) {
					queue.offer(p.getSpouse());
				}
			}

			return false;
		}

	}

	public static void main(String[] args) {

		MyPersonImplementation tester = new MyPersonImplementation();
		tester.testGrandChildToGreatGrandMotherRelationship(1);
		tester.testManToStrangerRelationship(3);
		tester.testGruncleRelationship(4);
		tester.testManToWifesCousinRelationship(5);
	}

	public void testManToWifesCousinRelationship(int testId) {
		// man
		Person jack = new MyPerson("Jack", 1, Person.MALE);

		// wife
		Person jill = new MyPerson("Jill", 2, Person.FEMALE);
		jack.setSpouse(jill);

		// wife's mother
		Person beth = new MyPerson("Beth", 3, Person.FEMALE);
		jill.addParent(beth);

		// wife's grandmother
		Person mary = new MyPerson("Mary", 4, Person.FEMALE);
		beth.addParent(mary);

		// wife's uncle
		Person dave = new MyPerson("Dave", 5, Person.MALE);
		mary.addChild(dave);

		// wife's aunt
		Person sally = new MyPerson("Sally", 6, Person.FEMALE);
		dave.setSpouse(sally);

		// wife's cousin
		Person andrew = new MyPerson("Andrew", 7, Person.MALE);
		sally.addChild(andrew);

		// wife's cousin's wife
		Person janet = new MyPerson("Janet", 8, Person.FEMALE);
		andrew.setSpouse(janet);

		// same person as janet...only related through SSN
		Person bigJ = new MyPerson("Janet", 8, Person.FEMALE);

		this.performTestAndPrintResults(testId, jack, janet, "Wife's Cousin",
				true);
		// test the reverse relationship
		this.performTestAndPrintResults(testId + 1, janet, jack,
				"Cousin's Husband", true);
		// test relationship through SSN
		this.performTestAndPrintResults(testId + 2, jack, bigJ,
				"With Janet being called by her nick name", true);

	}

	private void performTestAndPrintResults(int testCaseNumber, Person p1,
			Person p2, String relationship, boolean related) {

		String relatedOrNot = "not related";
		if (related) {
			relatedOrNot = "related";
		}

		System.out.print("TEST CASE #" + testCaseNumber);
		if (p1.isRelated(p2) == related) {
			System.out.println("(PASS): " + p1.getName() + " and "
					+ p2.getName() + "(" + relationship + ") are "
					+ relatedOrNot + ".\nYour code agrees!\n");
		} else {
			System.out.println("(FAIL): " + p1.getName() + " and "
					+ p2.getName() + "(" + relationship + ") are "
					+ relatedOrNot + ".\nYour code disagrees!\n");
		}

	}

	public void testManToStrangerRelationship(int testId) {

		Person jack = new MyPerson("Jack", 1, Person.MALE);
		Person stranger = new MyPerson("Perry", 2, Person.MALE);

		this.performTestAndPrintResults(testId, jack, stranger, "Stranger",
				false);

	}

	public void testGrandChildToGreatGrandMotherRelationship(int testId) {

		Person man = new MyPerson("Jack", 1, Person.MALE);

		// mother
		Person mother = new MyPerson("Jill", 2, Person.FEMALE);
		man.addParent(mother);

		// grandMother
		Person gm = new MyPerson("Beth", 3, Person.FEMALE);
		mother.addParent(gm);

		// great-grandMother
		Person ggm = new MyPerson("Mary", 4, Person.FEMALE);
		gm.addParent(ggm);

		// child
		Person child = new MyPerson("Peter", 5, Person.MALE);
		man.addChild(child);

		// grandChild
		Person grandChild = new MyPerson("Dave", 6, Person.MALE);
		child.addChild(grandChild);

		// is my grandChild related to my great-grandMother
		this.performTestAndPrintResults(testId, grandChild, ggm,
				"Great-Great-Great-GrandMa", true);
		this.performTestAndPrintResults(testId + 1, ggm, grandChild,
				"Great-Great-Great-GrandChild", true);
	}

	public void testGruncleRelationship(int testId) {

		// A gruncle is someone who is both a Grandfather & an Uncle to another
		// person

		// child
		Person child = new MyPerson("Billy", 0, Person.MALE);

		Person father = new MyPerson("Jack", 1, Person.MALE);
		Person mother = new MyPerson("Jill", 2, Person.FEMALE);
		father.setSpouse(mother);
		father.addChild(child);

		// father's father (grand father)
		Person grandPa = new MyPerson("Dave", 3, Person.MALE);
		father.addParent(grandPa);

		// mother's sister (aunt)
		Person gm = new MyPerson("Beth", 4, Person.FEMALE);
		mother.addParent(gm);
		Person aunt = new MyPerson("Shelly", 5, Person.FEMALE);
		gm.addChild(aunt);

		// father's father falls for mother's sister
		// (code to test looping due to double relationships)
		grandPa.setSpouse(aunt);

		Person gruncle = grandPa;
		// am i related to wife's uncle's wife?
		this.performTestAndPrintResults(testId, child, gruncle, "Gruncle", true);

	}

}
