
public final class Prob001 implements EulerSolution {

    public static void main(String[] args) {
	new Prob001().run();
    }

    public int sumBelow(int num) {
	int sum = 0;
	for (int i = 1; i < num; i ++) {
	    if (i % 3 == 0 || i % 5 == 0) {
		sum += i;
	    }
	}

	return sum;
    }

    public void run() {
	System.out.println(this.sumBelow(1000));
    }

}
