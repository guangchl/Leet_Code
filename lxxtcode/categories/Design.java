package categories;

import java.util.Iterator;

public class Design {

    /**
     * Peeking Iterator.
     *
     * Given an Iterator class interface with methods: next() and hasNext(),
     * design and implement a PeekingIterator that support the peek() operation
     * -- it essentially peek() at the element that will be returned by the next
     * call to next().
     */
    @tags.Design
    @tags.Source.LeetCode
    class PeekingIterator implements Iterator<Integer> {
        Integer buf = null;
        Iterator<Integer> iterator;

        public PeekingIterator(Iterator<Integer> iterator) {
            this.iterator = iterator;
        }

        // Returns the next element in the iteration without advancing the iterator.
        public Integer peek() {
            if (buf == null) {
                buf = iterator.next();
            }
            return buf;
        }

        // hasNext() and next() should behave the same as in the Iterator interface.
        // Override them if needed.
        @Override
        public Integer next() {
            if (buf != null) {
                Integer next = buf;
                buf = null;
                return next;
            } else {
                return iterator.next();
            }
        }

        @Override
        public boolean hasNext() {
            return buf != null || iterator.hasNext();
        }
    }

}
