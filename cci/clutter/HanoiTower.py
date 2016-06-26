import datetime

class HanoiTower:
    towers = []
    def __init__(self, n):
        # self.towers = []
        for i in xrange(3):
            self.towers.append([])
        for i in xrange(n, 0, -1):
            self.towers[0].append(i)

    def hanoiMove(self, fromTower, toTower, tempTower, n):
        if n == 1:
            self.towers[toTower].append(self.towers[fromTower].pop())
        else:
            self.hanoiMove(fromTower, tempTower, toTower, n - 1)
            self.hanoiMove(fromTower, toTower, tempTower, 1)
            self.hanoiMove(tempTower, toTower, fromTower, n - 1)
    
    def printTowers(self):
        for i in xrange(0, 3):
            print 'Tower', i+1, ':', self.towers[i]
        print ''
        
n = 26
hanoiTower = HanoiTower(n)
hanoiTower.printTowers()

starttime = datetime.datetime.now()
hanoiTower.hanoiMove(0, 2, 1, n)
endtime = datetime.datetime.now()

hanoiTower.printTowers()
print 'Time cost:', endtime-starttime

        
##    public HanoiTower(int n) {
##    towers = new ArrayList<Stack<Integer>>();
##    Stack<Integer> s1 = new Stack<Integer>();
##    Stack<Integer> s2 = new Stack<Integer>();
##    Stack<Integer> s3 = new Stack<Integer>();
##    for (int i = n; i > 0; i--) {
##    	s1.push(i);
##    }
##    towers.add(s1);
##    towers.add(s2);
##    towers.add(s3);
##    }
##
##	public static void main(String[] args) {
##    int n = 30;
##    HanoiTower ht = new HanoiTower(n);
##    ht.printTowers();
##    
##    long startTime = System.currentTimeMillis();
##    ht.hanoiMove(0, 2, 1, n);
##    long endTime = System.currentTimeMillis();
##    
##    ht.printTowers();
##    System.out.println("Time cost in milliseconds: " + (endTime-startTime));
##    // TODO Auto-generated method stub
