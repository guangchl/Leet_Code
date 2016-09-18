package clutter;

import java.util.ArrayList;

public class Fibonacci {
    /**
     * Recursion
     *
     * @param n
     * @return
     */
    public static int fib1(int n) {
        // base case
        if (n == 0) {
            return 0;
        } else if (n == 1) {
            return 1;
        }

        return fib1(n - 2) + fib1(n - 1);
    }

    public static int fib2(int n) {
        // base case
        if (n == 0) {
            return 0;
        } else if (n == 1) {
            return 1;
        }

        ArrayList<Integer> fibList = new ArrayList<Integer>();
        fibList.add(0);
        fibList.add(1);
        for (int i = 2; i <= n; i++) {
            fibList.add(fibList.get(i - 1) + fibList.get(i - 2));
        }

        return fibList.get(n);
    }

    public static int fib3(int n) {
        // base case
        if (n == 0) {
            return 0;
        } else if (n == 1) {
            return 1;
        }

        int[] prev = { 0, 1 };
        int current = 1;

        for (int i = 2; i < n; i++) {
            prev[0] = prev[1];
            prev[1] = current;
            current += prev[0];
        }

        return current;
    }

    public static int fib4(int n) {
        int a = 0;
        int b = 1;

        while (n-- > 0) {
            int temp = a + b;
            a = b;
            b = temp;
        }

        return a;
    }

    public static int fib5(int n) {
        double a = Math.sqrt(5);

        return (int) (1 / a * Math.pow((a + 1) / 2, n)
                - 1 / a * Math.pow((1 - a) / 2, n));
    }

    public static void main(String[] args) {
        int n = 10;

        for (int i = 0; i < n; i++) {
            System.out.print(fib1(i) + " ");
        }
        System.out.println();

        for (int i = 0; i < n; i++) {
            System.out.print(fib2(i) + " ");
        }
        System.out.println();

        for (int i = 0; i < n; i++) {
            System.out.print(fib3(i) + " ");
        }
        System.out.println();

        for (int i = 0; i < n; i++) {
            System.out.print(fib4(i) + " ");
        }
        System.out.println();

        for (int i = 0; i < n; i++) {
            System.out.print(fib5(i) + " ");
        }
    }

}
