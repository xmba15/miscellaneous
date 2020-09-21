
public final class Prob002 implements EulerSolution {

    public static void main(String[] args) {

	new Prob002().run();

    }

    public int evenFibSum(int num) {

	int sum = 0;
	int a = 1;
	int b = 2;

	while (a < num) {
	    if (a % 2 == 0) {
		sum += a;
	    }
	    int c = a + b;
	    a = b;
	    b = c;
	}

	return sum;

    }

    public void run() {

	System.out.println(this.evenFibSum(4000000));

    }

}
