package ninechapter;

public class BinarySearch {
	
	//******************************* TEMPLATE *******************************
	/**
	 * Template: Binary Search
	 */
    public int binarySearch(int[] nums, int target) {
    	if (nums == null || nums.length == 0) { // length == 0 is not necessary
    		return -1;
    	}
    	
    	int start = 0;
    	int end = nums.length - 1;
    	
    	while (start <= end) { // use "<" if search for a range
    		int mid = (start + end) >>> 1; // be aware of length of 2
    		
	    	if (nums[mid] == target) {
	            return mid;
	        } else if (nums[mid] < target) {
	        	start = mid + 1;
	        } else {
	        	end = mid - 1;
	        }
    	}
    	
    	return -1;
	}
    
    //******************************* PROBLEMS *******************************
    /**
     * 1. Search Insert Position
     */
    public int searchInsert(int[] A, int target) {
        if (A == null) {
            return -1;
        }

        int start = 0;
        int end = A.length - 1;
        
        while (start < end) {
            int mid = (start + end) >>> 1;
            
            if (A[mid] >= target) {
                end = mid;
            } else {
                start = mid + 1;
            }
        }
        
        return (A[start] < target) ? start + 1 : start;
    }
    
    /**
     * 2. Search for a Range
     */
    public int[] searchRange(int[] A, int target) {
    	int[] range = new int[2];
    	range[0] = -1;
    	range[1] = -1;
    	if (A == null) {
            return range;
        }

    	// search start
        int start = 0;
        int end = A.length - 1;
        while (start < end) {
            int mid = (start + end) >>> 1; // left middle
            
            if (A[mid] >= target) {
                end = mid;
            } else {
                start = mid + 1;
            }
        }
        
        // add valid start or return
        if (A[start] == target) {
        	range[0] = start;
        } else {
        	return range;
        }
        
        // search end
        end = A.length - 1;
        while (start < end) {
            int mid = (start + end + 1) >>> 1; // right middle
            
            if (A[mid] <= target) {
                start = mid;
            } else {
                end = mid - 1;
            }
        }
        
        // add end
       	range[1] = start;

       	return range;
    }
    
    /**
     * 3. 2774:木材加工
     * http://bailian.openjudge.cn/practice/2774
     */
//    #include <iostream>
//    using namespace std;
//
//    int cut_by_length(int wood[], int n, int length) {
//        int count = 0;
//        for (int i = 0; i < n; i++) {
//            count += wood[i] / length;
//        }
//        return count;
//    }
//
//    int main() {
//        int n, target, sum = 0;
//        
//        cin >> n >> target;
//        int wood[n];
//        for (int i = 0; i < n; i++) {
//            cin >> wood[i];
//            sum += wood[i];
//        }
//        
//        int start = 1, end = sum, mid, count;
//        while (start + 1 < end) {
//            mid = start + (end - start) / 2;
//            count = cut_by_length(wood, n, mid);
//            if (count == target) {
//                start = mid;
//            } else if (count < target) {
//                end = mid;
//            } else {
//                start = mid;
//            }
//        }
//        if (cut_by_length(wood, n, end) >= target) {
//            cout << end;
//        } else {
//            cout << start;
//        }
//        return 0;
//    }

    
    /**
	 * 4. Search in Rotated Sorted Array
	 * 
	 * It should be shortened by using same search without normal binary search
	 */
    public int search(int[] A, int target) {
        if (A == null || A.length == 0) {
            return -1;
        }
        
        int start = 0;
        int end = A.length - 1;
        
        while (start <= end) {
            int mid = (start + end) >>> 1;
            if (A[mid] == target) {
                return mid;
            } else if (A[mid] < target) {
                if (A[end] < A[mid]) {
                    start = mid + 1;
                } else if (A[end] >= target) {
                    return binarySearch(A, mid + 1, end, target);
                } else {
                    end = mid - 1;
                }
            } else if (A[mid] > target) {
                if (A[start] > A[mid]) {
                    end = mid - 1;
                } else if (A[start] <= target) {
                    return binarySearch(A, start, end - 1, target);
                } else {
                    start = mid + 1;
                }
            }
        }
        
        return - 1;
    }
    
    public int binarySearch(int[] A, int start, int end, int target) {
        while (start <= end) {
            int mid = (start + end) >>> 1;
            
            if (A[mid] == target) {
                return mid;
            } else if (A[mid] < target) {
                start = mid + 1;
            } else {
                end = mid - 1;
            }
        }
        
        return -1;
    }
    
	public int search2(int[] A, int target) {
		int start = 0;
		int end = A.length - 1;
		int mid;

		while (start + 1 < end) {
			mid = start + (end - start) / 2;
			if (A[mid] == target) {
				return mid;
			}
			if (A[start] < A[mid]) {
				// situation 1, red line
				if (A[start] <= target && target <= A[mid]) {
					end = mid;
				} else {
					start = mid;
				}
			} else {
				// situation 2, green line
				if (A[mid] <= target && target <= A[end]) {
					start = mid;
				} else {
					end = mid;
				}
			}
		} // while

		if (A[start] == target) {
			return start;
		}
		if (A[end] == target) {
			return end;
		}
		return -1;
	}
    
	/**
	 * Find Minimum in Rotated Sorted Array
	 * You may assume no duplicate exists in the array.
	 */
	public int findMin(int[] num) {
		if (num == null || num.length == 0) {
			return - 1;
		}
		
        int left = 0;
        int right = num.length - 1;
        
        while (left < right) {
            int mid = (left + right) >> 1;
            if (num[mid] < num[right]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        
        return num[left];
    }
	
    /**
     * 5. Search a 2D Matrix
     */
    public boolean searchMatrix(int[][] matrix, int target) {
    	if (matrix == null || matrix.length == 0 || matrix[0].length== 0) {
            return false;
        }
        
        int m = matrix.length;
        int n = matrix[0].length;
        int start = 0;
        int end = m * n - 1;
        
        while (start <= end) {
            int mid = (start + end) >>> 1;
            
            int row = mid / n;
            int column = mid % n;
            if (matrix[row][column] == target) {
                return true;
            } else if (matrix[row][column] < target) {
                start = mid + 1;
            } else {
                end = mid - 1;
            }
        }
        
        return false;
    }
    
    /**
	 * 6. Find Peek
	 * 
	 * 查找峰值。假设有一个数组,相邻两数都不相等，且A[0]<A[1], A[length-2] > A[length-1] 。如果A[i] >
	 * A[i-1] && A[i] > A[i+1]，那么认为A[i]是一个峰值。数组中可能存在多个峰值。给定A，找到任意一个峰值。
	 */
    public int findPeakValue(int[] num) {
        if (num == null || num.length < 3) {
            return -1;
        }

        int beg = 1;
        int end = num.length - 2;
        int mid;

        while (beg + 1 < end) {
            mid = beg + (end - beg) / 2;
            if (num[mid - 1] < num[mid] && num[mid] > num[mid + 1]) {
                return mid;
            } else if (num[mid] < num[mid + 1]) {
                beg = mid;
            } else {
                end = mid;
            }
        }

        if (num[beg - 1] < num[beg] && num[beg] > num[beg + 1]) {
            return beg;
        }

        if (num[end - 1] < num[end] && num[end] > num[end + 1]) {
            return end;
        }

        return -1;
    }
    
    /**
     * 
     */
  //Solution to copy books, the following code is in c++ since wikioi 
  //only accetps c++
//  #include <stdio.h>
//  #include <string.h>
//  #include <stdlib.h>
//  #include <algorithm>
//  #include <iostream>
//  #include <cstdio>
//  using namespace std;
//
//  // Check whether a given number of pages in a slice is
//  // valid, i.e. all the books could get copied.
//  bool isValid(int M, int K, int* pages, int sliceNum) {
//      int curSliceNum = 0;
//      int curBook = M - 1;
//      for(int i = K - 1; i >= 0; i--) {
//          curSliceNum = 0;
//
//          while(curSliceNum + pages[curBook] <= sliceNum && 
//                  curBook >= 0) {
//              curSliceNum += pages[curBook];
//              curBook--;
//          }
//
//          if (curBook < 0) {
//              return true;
//          }
//      }
//
//      return false;
//  }
//
//
//  // Use binary search to find the optimal number of pages in a slice.
//  int search(int M, int K, int* pages, int minSliceNum, int maxSliceNum) {
//      int beg = minSliceNum;
//      int end = maxSliceNum;
//      int mid;
//
//      while (beg + 1 < end) {
//          mid = (beg + end) / 2;
//          if (isValid(M, K, pages, mid)) {
//              end = mid;
//          } else {
//              beg = mid;
//          }
//      }
//
//      if (isValid(M, K, pages, end)) {
//          return end;
//      }
//
//      return beg;
//  }
//
//  int main() {
//      int M, K;
//      scanf("%d %d", &M, &K);
//
//      int* pages = new int[M];
//      int* startBook = new int[K];
//      int* endBook = new int[K];
//      int maxSliceNum = 0;
//      int minSliceNum = 0;
//      int optimalSliceNum;
//      for(int i = 0; i < M; i++) {
//          scanf("%d ", &pages[i]);
//          minSliceNum = min(pages[i], minSliceNum);
//          maxSliceNum += pages[i];
//      }
//
//      optimalSliceNum = search(M, K, pages, minSliceNum, maxSliceNum);
//
//      int curSliceNum = 0;
//      int curBook = M - 1;
//      for(int i = K - 1; i >= 0; i--) {
//          curSliceNum = 0;
//          endBook[i] = curBook;
//          while (curSliceNum + pages[curBook] <= optimalSliceNum && 
//                  curBook >= i) {
//              curSliceNum += pages[curBook];
//              curBook--;
//          }
//          startBook[i] = curBook + 1;
//      }
//
//      for(int i = 0; i < K; i++) {
//          printf("%d %d\n", startBook[i] + 1, endBook[i] + 1);
//      }
//
//      delete [] endBook; 
//      delete [] startBook; 
//      delete [] pages; 
//
//      return 0;
//  }
    
	public void test() {
		int[] nums = new int[3];
		nums[0] = 1;
		nums[1] = 2;
		nums[2] = 3;
		System.out.println(binarySearch(nums, 1));
	}
	
	public static void main(String[] args) {
		BinarySearch test = new BinarySearch();
		test.test();
	}

}
