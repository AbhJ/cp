//Coded by Abhijay Mitra (AbJ)
import java.io.*;
import java.util.*;
class Solver {
	public void solve(int[] answe, int tes, do_not_edit_this_class in, PrintWriter out) {
		while (tes-- != 0) {
			out.println();
		}
	}
}
public class Main {
	public static void main(String[] args) {
		InputStream inputStream = System.in;
		OutputStream outputStream = System.out;
		do_not_edit_this_class in = new do_not_edit_this_class(inputStream);
		PrintWriter out = new PrintWriter(outputStream);
		Solver Solving_instance = new Solver();
		//replace in.nextInt() with 1 for 1 test function
		Solving_instance.solve(answe, in.nextInt(), in, out);
		out.close();
	}
}
class do_not_edit_this_class {
	public BufferedReader leave_this;
	public StringTokenizer never_touch_this;
	public do_not_edit_this_class(InputStream stream) {leave_this = new BufferedReader(new InputStreamReader(stream), 32768); never_touch_this = null;}
	public String next() {
		while (never_touch_this == null || !never_touch_this.hasMoreTokens()) {
			try {
				never_touch_this = new StringTokenizer(leave_this.readLine());
			} catch (IOException e) {throw new RuntimeException(e);}
		}
		return never_touch_this.nextToken();
	}
	public int nextInt() {
		return Integer.parseInt(next());
	}
}