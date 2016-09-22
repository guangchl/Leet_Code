package categories;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.Test;

/**
 * Design questions.
 *
 * @author guangcheng
 */
public class Design {

    /**
     * Peeking Iterator.
     *
     * Given an Iterator class interface with methods: next() and hasNext(),
     * design and implement a PeekingIterator that support the peek() operation
     * -- it essentially peek() at the element that will be returned by the next
     * call to next().
     *
     * Follow up: How would you extend your design to be generic and work with
     * all types, not just integer?
     */
    @tags.Design
    @tags.Company.Amazon
    @tags.Company.Google
    @tags.Company.Yahoo
    @tags.Source.LeetCode
    @tags.Status.Easy
    class PeekingIterator implements Iterator<Integer> {
        Integer buf = null;
        Iterator<Integer> iterator;

        public PeekingIterator(Iterator<Integer> iterator) {
            this.iterator = iterator;
        }

        // Returns the next element in the iteration without advancing the
        // iterator.
        public Integer peek() {
            if (buf == null) {
                buf = iterator.next();
            }
            return buf;
        }

        // hasNext() and next() should behave the same as in the Iterator
        // interface.
        // Override them if needed.
        @Override
        public Integer next() {
            if (buf != null) {
                Integer next = buf;
                buf = null;
                return next;
            }
            return iterator.next();
        }

        @Override
        public boolean hasNext() {
            return buf != null || iterator.hasNext();
        }
    }

    /**
     * Unique Word Abbreviation.
     *
     * An abbreviation of a word follows the form <first letter><number><last
     * letter>. Below are some examples of word abbreviations:
     *
     * a) it --> it (no abbreviation)
     *
     * b) d|o|g --> d1g
     *
     * c) i|nternationalizatio|n --> i18n
     *
     * d) l|ocalizatio|n --> l10n
     *
     * Assume you have a dictionary and given a word, find whether its
     * abbreviation is unique in the dictionary. A word's abbreviation is unique
     * if no other word from the dictionary has the same abbreviation.
     *
     * Example: Given dictionary = [ "deer", "door", "cake", "card" ],
     * isUnique("dear") -> false, isUnique("cart") -> true, isUnique("cane") ->
     * false, isUnique("make") -> true.
     */
    @tags.HashTable
    @tags.Design
    @tags.Company.Google
    @tags.Status.NeedPractice
    public class ValidWordAbbr {
        // Your ValidWordAbbr object will be instantiated and called as such:
        // ValidWordAbbr vwa = new ValidWordAbbr(dictionary);
        // vwa.isUnique("Word");
        // vwa.isUnique("anotherWord");

        Map<String, Set<String>> abbrSets = new HashMap<>();

        public ValidWordAbbr(String[] dictionary) {
            for (String word : dictionary) {
                String abbr = getAbbr(word);

                if (abbrSets.containsKey(abbr)) {
                    abbrSets.get(abbr).add(word);
                } else {
                    Set<String> set = new HashSet<>();
                    set.add(word);
                    abbrSets.put(abbr, set);
                }
            }
        }

        public boolean isUnique(String word) {
            String abbr = getAbbr(word);
            if (abbrSets.containsKey(abbr)) {
                Set<String> set = abbrSets.get(abbr);
                return set.size() == 1 && set.contains(word);
            }
            return true;
        }

        private String getAbbr(String word) {
            if (word.length() <= 2) {
                return word;
            }
            StringBuilder sb = new StringBuilder();
            sb.append(word.charAt(0));
            sb.append(word.length() - 2);
            sb.append(word.charAt(word.length() - 1));
            return sb.toString();
        }
    }

    /**
     * Design Hit Counter.
     *
     * Design a hit counter which counts the number of hits received in the past
     * 5 minutes. Each function accepts a timestamp parameter (in seconds
     * granularity) and you may assume that calls are being made to the system
     * in chronological order (ie, the timestamp is monotonically increasing).
     * You may assume that the earliest timestamp starts at 1. It is possible
     * that several hits arrive roughly at the same time.
     *
     * Example: HitCounter counter = new HitCounter();
     *
     * // hit at timestamp 1. counter.hit(1);
     *
     * // hit at timestamp 2. counter.hit(2);
     *
     * // hit at timestamp 3. counter.hit(3);
     *
     * // get hits at timestamp 4, should return 3. counter.getHits(4);
     *
     * // hit at timestamp 300. counter.hit(300);
     *
     * // get hits at timestamp 300, should return 4. counter.getHits(300);
     *
     * // get hits at timestamp 301, should return 3. counter.getHits(301);
     *
     * Follow up: What if the number of hits per second could be very large?
     * Does your design scale? What if timestamp granularity is infinite small?
     */
    @tags.Design
    @tags.Company.Google
    @tags.Company.Dropbox
    @tags.Status.NeedPractice
    public class HitCounter {

        /**
         * Your HitCounter object will be instantiated and called as such:
         * HitCounter obj = new HitCounter();
         * obj.hit(timestamp);
         * int param_2 = obj.getHits(timestamp);
         */

        class Hit {
            int timestamp;
            int count = 1;

            public Hit(int timestamp) {
                this.timestamp = timestamp;
            }
        }

        private LinkedList<Hit> hits = new LinkedList<>();
        private int size = 0;

        /** Initialize your data structure here. */
        public HitCounter() {
        }

        /**
         * Record a hit.
         *
         * @param timestamp
         *            - The current timestamp (in seconds granularity).
         */
        public void hit(int timestamp) {
            if (hits.isEmpty() || hits.getLast().timestamp != timestamp) {
                Hit newLast = new Hit(timestamp);
                hits.addLast(newLast);
            } else {
                hits.getLast().count += 1;
            }
            size += 1;
        }

        /**
         * Return the number of hits in the past 5 minutes.
         *
         * @param timestamp
         *            - The current timestamp (in seconds granularity).
         */
        public int getHits(int timestamp) {
            while (!hits.isEmpty()
                    && hits.getFirst().timestamp <= timestamp - 300) {
                size -= hits.removeFirst().count;
            }
            return size;
        }
    }

    // ---------------------------------------------------------------------- //
    // ----------------------------- Hashtable ------------------------------ //
    // ---------------------------------------------------------------------- //

    /**
     * Implement HashMap.
     *
     * @param <K>
     * @param <V>
     */
    public class SimpleHashMap<K, V> {
        class Element {
            K key;
            V val;

            public Element(K key, V val) {
                this.key = key;
                this.val = val;
            }
        }

        public ArrayList<List<Element>> array;

        public SimpleHashMap() {
            array = new ArrayList<>();
        }

        public void put(K key, V val) {
            int hash = key.hashCode() % array.size();
            List<Element> list = array.get(hash);
            if (list == null) {
                list = new ArrayList<>();
                array.set(hash, list);
            }
            for (int i = 0; i < list.size(); i++) {
                Element elm = list.get(i);
                if (elm.key == key) {
                    elm.val = val;
                    return;
                }
            }
            list.add(new Element(key, val));
        }

        public V get(K key) {
            int hash = key.hashCode() % array.size();
            List<Element> list = array.get(hash);
            for (int i = 0; i < list.size(); i++) {
                if (list.get(i).key == key) {
                    return list.get(i).val;
                }
            }
            return null;
        }
    }

    /**
     * Implementing join semantics.
     */
    class JoinThread {
        private static final long SLEEP_INTERVAL_MS = 1000;
        private boolean running = true;

        public void start() {
            Thread thread = new Thread(new Runnable() {
                @Override
                public void run() {
                    System.out.println("Hello world.");
                    try {
                        Thread.sleep(SLEEP_INTERVAL_MS);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }

                    synchronized (JoinThread.this) { // NOTE: cannot be this
                        running = false;
                        JoinThread.this.notify();
                    }
                }
            });
            thread.start();
        }

        public void join() throws InterruptedException {
            synchronized (this) {
                while (running) {
                    System.out.println("Waiting for peer threads to finish.");
                    wait();
                }
                System.out.println("Peer threads finished.");
            }
        }

    }

    // ---------------------------------------------------------------------- //
    // --------------------------- Blocking Queue --------------------------- //
    // ---------------------------------------------------------------------- //

    public class SimpleBlockingQueue<T> {
        private Queue<T> queue;
        private int max;

        public SimpleBlockingQueue(int num) {
            this.max = num;
            queue = new LinkedList<T>();
        }

        public int getSize() {
            return queue.size();
        }

        public void offer(T task) {
            synchronized (queue) {
                // cannot use if statement here
                // http://stackoverflow.com/questions/1038007/why-should-wait-always-be-called-inside-a-loop
                while (queue.size() == max) {
                    try {
                        queue.wait();
                    } catch (InterruptedException e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();
                    }
                }
                queue.offer(task);
                queue.notify();
            }
        }

        public T poll() {
            synchronized (queue) {
                while (queue.size() == 0) {
                    try {
                        queue.wait();
                    } catch (InterruptedException e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();
                    }
                }
                T task = queue.poll();
                queue.notify();
                return task;
            }
        }
    }

    /**
     * <p>
     * This class and its iterator implement all of the <em>optional</em>
     * methods of the {@link ArrayBlockingQueue} and {@link BlockingQueue}
     * interfaces.
     *
     * @param <T>
     */
    class SimpleArrayBlockingQueue<T> {
        /** The queued items */
        final Object[] items;

        /** items index for next take, poll, peek or remove */
        int takeIndex;

        /** items index for next put, offer, or add */
        int putIndex;

        /** Number of elements in the queue */
        int count;

        /*
         * Concurrency control uses the classic two-condition algorithm found in
         * any textbook.
         */

        /** Main lock guarding all access */
        // final ReentrantLock lock;

        /** Condition for waiting takes */
        // private final Condition notEmpty;

        /** Condition for waiting puts */
        // private final Condition notFull;

        /*
         * Constructors.
         */

        /**
         * Creates an {@code SimpleArrayBlockingQueue} with the given (fixed)
         * capacity and default access policy.
         *
         * @param capacity
         *            the capacity of this queue
         * @throws IllegalArgumentException
         *             if {@code capacity < 1}
         */
        public SimpleArrayBlockingQueue(int capacity) {
            this(capacity, false);
        }

        /**
         * Creates an {@code ArrayBlockingQueue} with the given (fixed) capacity
         * and the specified access policy.
         *
         * @param capacity
         *            the capacity of this queue
         * @param fair
         *            if {@code true} then queue accesses for threads blocked on
         *            insertion or removal, are processed in FIFO order; if
         *            {@code false} the access order is unspecified.
         * @throws IllegalArgumentException
         *             if {@code capacity < 1}
         */
        public SimpleArrayBlockingQueue(int capacity, boolean fair) {
            if (capacity <= 0)
                throw new IllegalArgumentException();
            this.items = new Object[capacity];
            // lock = new ReentrantLock(fair);
            // notEmpty = lock.newCondition();
            // notFull = lock.newCondition();
        }

        /*
         * Methods.
         */

        public synchronized void enqueue(T x) {
            // assert lock.getHoldCount() == 1;
            // assert items[putIndex] == null;

            // notEmpty.signal();

            while (count == items.length) {
                try {
                    this.wait();
                } catch (InterruptedException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }

            final Object[] items = this.items;
            items[putIndex] = x;
            if (++putIndex == items.length)
                putIndex = 0;
            count++;
            this.notify();
        }

        public synchronized T dequeue() {
            while (count == 0) {
                try {
                    wait();
                } catch (InterruptedException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }

            @SuppressWarnings("unchecked")
            T elem = (T) items[takeIndex];
            count--;
            this.notify();
            return elem;
        }
    }

    // ---------------------------------------------------------------------- //
    // -------------------------- Multi-threading --------------------------- //
    // ---------------------------------------------------------------------- //

    /**
     * Reader writer lock.
     */
    public class ReaderWriterLock {
        private int readers;
        private boolean writer;
        public final ReadLock readLock;
        public final WriteLock writeLock;

        public ReaderWriterLock() {
            readers = 0;
            writer = false;
            readLock = new ReadLock();
            writeLock = new WriteLock();
        }

        class ReadLock {
            public void lock() {
                synchronized (ReaderWriterLock.this) {
                    try {
                        while (writer) {
                            ReaderWriterLock.this.wait();
                        }
                        readers++;
                    } catch (InterruptedException e) {
                        // error handling
                    }
                }
            }

            public void unlock() {
                synchronized (ReaderWriterLock.this) {
                    readers--;
                    if (readers == 0) {
                        ReaderWriterLock.this.notifyAll();
                    }
                }
            }
        }

        class WriteLock {
            public void lock() {
                synchronized (ReaderWriterLock.this) {
                    try {
                        while (readers > 0) {
                            ReaderWriterLock.this.wait();
                        }
                        writer = true;
                    } catch (InterruptedException e) {
                        // error handling
                    }
                }
            }

            public void unlock() {
                synchronized (ReaderWriterLock.this) {
                    writer = false;
                    ReaderWriterLock.this.notifyAll();
                }
            }
        }
    }

    /**
     * Delayed task scheduler.
     *
     * General guideline: check all conditions when notified and before do any
     * task with any condition assumption.
     */
    public class SimpleScheduler {
        private class TaskRunner implements Runnable {
            @Override
            public void run() {
                // running probably can be put inside of synchronized, because
                // there is nothing in between. And this implementation cannot
                // make sure all running tasks finish.
                while (running) {
                    synchronized (SimpleScheduler.this) {
                        try {
                            while (tasks.isEmpty()) {
                                // running can change here and we should return
                                // since there is nothing to run. And we want to
                                // abandon all queued task anyway which is one
                                // of our choices here.
                                if (!running) {
                                    return;
                                }
                                SimpleScheduler.this.wait();
                            }
                            long now = System.currentTimeMillis();
                            Task t = tasks.peek();
                            if (t.getTimeToRun() < now) {
                                tasks.poll();
                                t.run();
                            } else {
                                // running can change here and we should return
                                // by next while condition check since we want
                                // to abandon all queued task anyway which is
                                // one of our choices here.
                                SimpleScheduler.this
                                        .wait(t.getTimeToRun() - now);
                            }
                        } catch (InterruptedException e) {
                            // TODO: error handling
                            Thread.currentThread().interrupt();
                        }
                    }
                }
            }
        }

        private PriorityQueue<Task> tasks;
        private final Thread taskRunnerThread;
        private volatile boolean running;
        private final AtomicInteger taskId;

        public SimpleScheduler() {
            tasks = new PriorityQueue<>();
            taskRunnerThread = new Thread(new TaskRunner());
            running = true;
            taskId = new AtomicInteger(0);
            taskRunnerThread.start();
        }

        public void schedule(Task task, long delayMs) {
            long timeToRun = System.currentTimeMillis() + delayMs;
            task.setTimeToRun(timeToRun);
            task.setId(taskId.incrementAndGet());
            synchronized (this) {
                tasks.offer(task);
                this.notify();
            }
        }

        public void stop() throws InterruptedException {
            synchronized (this) {
                // Since running is protected by lock, only place inside of lock
                // it could change is after wait(). Any shared variable requires
                // carefully check what could happen if wait and notify happens.
                running = false;
                this.notify();
            }
            taskRunnerThread.join();
        }
    }

    public class Task implements Comparable<Task> {
        private int id;
        private long timeToRun;

        public void setTimeToRun(long timeToRun) {
            this.timeToRun = timeToRun;
        }

        public long getTimeToRun() {
            return timeToRun;
        }

        public void setId(int id) {
            this.id = id;
        }

        public int getId() {
            return id;
        }

        public void run() {
            // Do the task.
        }

        @Override
        public int compareTo(Task t) {
            return (int) (timeToRun - t.timeToRun);
        }
    }

    @Test
    public void test() throws InterruptedException {
        joinThreadTest();
    }

    private void joinThreadTest() throws InterruptedException {
        JoinThread jt = new JoinThread();
        jt.start();
        jt.join();
    }
}
