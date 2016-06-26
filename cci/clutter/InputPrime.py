from math import sqrt
import sys

def isPrime(n):
    if n == 1:
        return False
    elif n == 2:
        return True
    elif n % 2 == 0:
        return False
    
    for i in xrange(3, int(sqrt(n)) + 1, 2):
        if n % i == 0:
            return False

    return True


# check if a input is prime integer
def inputPrime():
    while(True):
        number = raw_input("Please enter a valid integer: ")
        try:
            number = int(number)
            break
        except:
            pass
        
    if isPrime(number):
        return 'yes'
    else:
        return 'no'


# Write a command-line program that reads 16-bit integers from stdin
# prints "yes" or "no" depending on whether each number is a prime
# and loops while it can still read more numbers from stdin.
# Numbers will be separated by whitespace
# and there may be an infinite sequence of them
# (e.g. the input could be coming from some other program).
# Any input errors should cause warnings to be printed on stderr.
# The program needs to run quickly, and produce accurate results.
def inputPrimes():
    integers = raw_input("Enter integers: ")
    numbers = integers.split()

    for number in numbers:
        try:
            number = int(number)
            if isPrime(number):
                print 'yes '
            else:
                print 'no '
        except:
            sys.stderr.write('Error: ' + number + ' is not a valid integer.\n')


#print inputPrime()
inputPrimes()
