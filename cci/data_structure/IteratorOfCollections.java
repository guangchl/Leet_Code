package data_structure;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

//shankarvn@gmail.com

/**
 * Guangcheng Lu
 * 
 * Here I use coc (Collection of Collections)
 */
public class IteratorOfCollections {

	public static void main(String[] args) {
        List<String> list1 = new ArrayList<String>();
        list1.add("foo1");
        list1.add("foo2");

        List<String> list2 = new ArrayList<String>();
        list2.add("foo3");
        list2.add("foo4");

        List<List<String>> myCollection = new ArrayList<List<String>>();
        myCollection.add(list1);
        myCollection.add(list2);

        CustomIterator<String> iterator = new CustomIterator<String>(myCollection);
        while(iterator.hasNext()){
            System.out.println(iterator.next());
        }

    }
	
	public static class CustomIterator<T> implements Iterator<T>{
		// iterate through all collections in the outer coc
	    private Iterator<? extends Collection<T>> outerIter;
	    // iterate through a specific collection in the coc
	    private Iterator<T> innerIter;
	    // real next element in coc
	    private T nextElem;
	    // flag show if CustomIterator reach its end
	    private boolean reachEnd = false;

	    public CustomIterator(Collection<? extends Collection<T>> coc) {
	    	// get outer iterator
	        outerIter = coc.iterator();
	        // search the next function and set the iterators to right position
	        findNext();
	    }

	    /**
	     * Helper function that used by next() internally
	     * 
	     * find the next element in the coc
	     */
	    private void findNext() {
	    	// find proper outer iterator
	        do {
	            if (innerIter == null || !innerIter.hasNext()) {
	                if (!outerIter.hasNext()) {
	                    reachEnd = true;
	                    return;
	                } else
	                    innerIter = outerIter.next().iterator();
	            }
	        } while (!innerIter.hasNext());

	        // set the next element
	        nextElem = innerIter.next();
	    }

	    /**
	     * check if CustomIterator reach its end
	     */
	    @Override
	    public boolean hasNext() {
	        return !reachEnd;
	    }

	    /**
	     * @return the next element in coc
	     */
	    @Override
	    public T next() {
	        if (reachEnd) {
	            throw new NoSuchElementException();
	        }
	        
	        // if no exception, find next and return
	        T next = nextElem;
	        findNext();
	        return next;
	    }

	    @Override
	    public void remove() {
	        //TODO
	    }

	}
}
