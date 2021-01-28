/*
 * ARMAX(0,0,1)-GARCH(1,1) MLE routine 
 * 
 * This routine implements the Maximum Likelihood Estimation (MLE) for 
 * ARMAX(0,0,1)-GARCH(1,1) process described as below
 * 
 * x, y are two time series:     y(t) = kappa + gamma * x(t) + e(t)
 * e is an residual series:      e(t) ~ Normal(0, h(t))
 * variance for residual series: h(t) = omega + alpha * e(t-1)^2 + beta * h(t-1)
 * 
 * 	kappa: intercept of linear regression
 * 	gamma: slope of linear regression
 * 	omega: conditional mean variance
 * 	alpha: coefficients related to lagged-1 residuals
 * 	beta : coefficient related to lagged-1 conditional variances
 * 
 * x, y series are assumed to follow a simple linear regression (e.g. ARMAX(0,0,1))
 * e series is assumed to follow a GARCH(1,1) process
 * Three distributions are provided for the e series:
 * 
 * 1. Normal Distribution, 
 * 2. Student T Distribution
 * 3. Exponential Power Distribution
 *  
 * The BHHH method is used to solve MLE problem, a nonliear optimization problem. 
 * BHHH method is a variant of Gauss-Newton method.   
 * Finite difference method is used for evaluating the Jacobian Matrix.
 * For a camparison, the MLE is also calculated by Nelder Mead Simplex method.
 * The Correctness of the BHHH method implementation can be justified by   
 * comparing the estimates from the two optimization methods.
 * 
 * Additional Thirdparty Open source packages are required for the program:
 * 
 * Open-source CERN/COLT Java Package can be found here:
 * http://acs.lbl.gov/software/colt/
 *
 * Flanagan's Scientific Java Package can be found here:
 * http://www.ee.ucl.ac.uk/~mflanaga/java/ 
 *
 * (C) Copyright 2010, Changwei Xiong (axcw@hotmail.com)
 * 
 */

import java.lang.Math;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.stat.Gamma;
import cern.jet.random.engine.RandomEngine;
import cern.jet.random.Uniform;
import cern.jet.random.Normal;
import cern.jet.random.StudentT;
import cern.jet.random.ExponentialPower;

import flanagan.math.Maximisation;
import flanagan.math.MaximisationFunction;

public class EstimateGarchRegress {
	/*
	 * Random Number Generator Interface for various distributions
	 */
	static interface RNG {
		double nextValue();
	}

	/*
	 * Random Number Generator Class for Normal distribution
	 */
	static class RNG_Normal extends Normal implements RNG {
		public RNG_Normal() {
			super(0, 1, RandomEngine.makeDefault());
			// TODO Auto-generated constructor stub
		}

		@Override
		public double nextValue() {
			return this.nextDouble();
		}
	}

	/*
	 * Random Number Generator Class for Student t distribution
	 */
	static class RNG_StudT extends StudentT implements RNG {
		RNG_StudT(double df) {// degree of freedom
			super(df, RandomEngine.makeDefault());
		}

		@Override
		public double nextValue() {
			return this.nextDouble();
		}
	}

	/*
	 * Random Number Generator Class for Exponential Power distribution a.k.a,
	 * Generalized Gaussian distribution, Generalized Error distribution
	 */
	static class RNG_ExpPow extends ExponentialPower implements RNG {
		RNG_ExpPow(double tau) {// parameter for distribution shape, 2 for
								// Gaussian
			super(tau, RandomEngine.makeDefault());
		}

		@Override
		public double nextValue() {
			return this.nextDouble();
		}
	}

	/*
	 * Generate simulated time series for testing purpose
	 */
	static double[][] GenGarchTimeSeries(double[] param, final int T, RNG rng) {
		final cern.jet.math.Functions Fn = cern.jet.math.Functions.functions;
		final DoubleFactory1D FacV = DoubleFactory1D.dense;

		double kap = param[0];
		double gam = param[1];
		double ome = param[2];
		double alp = param[3];
		double bet = param[4];
		double tau = param[5]; // distribution parameter for StudT and ExpPow

		DoubleMatrix1D h = FacV.make(T, 0.0);
		DoubleMatrix1D e = FacV.make(T, 0.0);
		h.set(0, ome / (1 - alp - bet));
		for (int i = 1; i < T; i++) {
			int k = i - 1;
			double ek = e.get(k);
			double hi = ome + alp * ek * ek + bet * h.get(k);
			h.set(i, hi);
			e.set(i, Math.sqrt(hi) * rng.nextValue());
		}

		double[] xval = new double[T];
		for (int i = 0; i < T; i++) {
			xval[i] = rng.nextValue();
		}
		DoubleMatrix1D x = FacV.make(xval);
		DoubleMatrix1D y = e.assign(Fn.plus(kap)).assign(x, Fn.plusMult(gam));
		return new double[][] { x.toArray(), y.toArray() };
	}

	/*
	 * Randomly generate parameters for simulated testing time series the MLE
	 * routine is going to estimate these parameters the greater the estimates
	 * are close to these values, the better the estimation is achieved
	 */
	static double[] GenParamArray() {
		Uniform RndUnif = new Uniform(RandomEngine.makeDefault());
		// kappa: intercept of linear regression
		double kappa = RndUnif.nextDoubleFromTo(-0.1, 0.1);
		// gamma: slope of linear regression
		double gamma = RndUnif.nextDoubleFromTo(0.7, 1.3);
		// omega: conditional mean variance
		double omega = RndUnif.nextDoubleFromTo(0.1, 0.3);
		// alpha: coefficients related to lagged-1 innovations
		double alpha = RndUnif.nextDoubleFromTo(0.1, 0.5);
		// beta: coefficient related to lagged-1 conditional variance
		double beta = 0.7 - alpha;
		// Tau: additional parameter different distributions
		double tau = RndUnif.nextDoubleFromTo(1.25, 8);
		return new double[] { kappa, gamma, omega, alpha, beta, tau };
	}

	
	static String DoubleToString(double d){
		final int n = 8;
		String s = Double.toString(d);
		int l = s.length();
		s = s.substring(0, Math.min(l, n));
		s = s + "         ".substring(0, Math.max(n-l,0));		
		return s;
	}
		
	static void ShowOutput(double[] param, GarchRegress garchreg){
		String pname[] = { "kappa: ", "gamma: ", "omega: ", "alpha: ", "beta : ", "tau  : " };
		double[] param_BHHH = garchreg.Calc_MLE_by_BHHH();
		double[] param_NM = garchreg.Calc_MLE_by_NelderMead();
		System.out.println("---------------------------------------------------------");
		System.out.println("Parameter" + "\t" 
				+ "Orig. Value" + "\t" 
				+ "Esti.(BHHH)" + "\t" 
				+ "Esti.(NelderMead)");
		for (int i = 0; i < param_BHHH.length; i++) {
			System.out.println(pname[i] + "\t\t" 
					+ DoubleToString(param[i]) + "\t"
					+ DoubleToString(param_BHHH[i]) + "\t"
					+ DoubleToString(param_NM[i]));
		}
		System.out.println("---------------------------------------------------------\n\n");
	}
	
	/**
	 ****************************************** 
	 ****************************************** 
	 * Here is the MAIN procedure to start. *
	 ****************************************** 
	 ****************************************** 
	 */
	public static void main(String[] args) {
		double param[] = GenParamArray();
		
		System.out.println("Normal Distribution:");
		double data_normal[][] = GenGarchTimeSeries(param, 5000, new RNG_Normal());
		ShowOutput(param, new GarchRegress_Normal(data_normal[0], data_normal[1]));		
		
		System.out.println("Student T Distribution:");
		double data_studt[][] = GenGarchTimeSeries(param, 5000, new RNG_StudT(param[5]));
		ShowOutput(param, new GarchRegress_StudT(data_studt[0], data_studt[1]));
		
		System.out.println("Exponential Power Distribution:");
		double data_exppow[][] = GenGarchTimeSeries(param, 5000, new RNG_ExpPow(param[5]));
		ShowOutput(param, new GarchRegress_ExpPow(data_exppow[0], data_exppow[1]));
	}
}

/*
 * abstract Garch(1,1)-Regression class defining interface and implementing
 * common methods
 */
abstract class GarchRegress {
	final protected static cern.jet.math.Functions Fn = cern.jet.math.Functions.functions;
	final protected static DoubleFactory1D FacV = DoubleFactory1D.dense;
	final protected static DoubleFactory2D FacM = DoubleFactory2D.dense;
	final protected static Algebra Alg = new Algebra();
	final protected static double[] Step = { 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4 };

	protected DoubleMatrix1D x = null;
	protected DoubleMatrix1D y = null;
	protected int T = 0;// length of x and y time series
	protected int N = 0;// number of parameters

	GarchRegress(double[] x, double[] y) {
		// do some pre-check here !!!
		this.T = x.length;
		this.x = FacV.make(x);
		this.y = FacV.make(y);
	}

	/*
	 * calculate e and h series for likelihood calculation
	 */
	protected DoubleMatrix1D[] Calc_e_and_h(final DoubleMatrix1D param) {
		double kappa = param.get(0);
		double gamma = param.get(1);
		double omega = param.get(2);
		double alpha = param.get(3);
		double beta = param.get(4);

		DoubleMatrix1D h = FacV.make(T);
		DoubleMatrix1D e = y.copy().assign(Fn.minus(kappa)).assign(x, Fn.minusMult(gamma));

		h.set(0, omega / (1.0 - alpha - beta));
		for (int i = 1; i < T; i++) {
			double eval = e.get(i - 1);
			h.set(i, omega + alpha * eval * eval + beta * h.get(i - 1));
		}
		return new DoubleMatrix1D[] { e, h };
	}

	/*
	 * BacktrackingLineSearch for optimization routine
	 */
	private double BacktrackingLineSearch(final DoubleMatrix1D param, final DoubleMatrix1D dp,
			final DoubleMatrix1D J) {
		final double a = 0.2;
		final double b = 0.6;
		double steplength = 1;
		while (true) {
			double fnew = LogLikelihood(param.copy().assign(dp, Fn.plusMult(steplength)));
			double fold = LogLikelihood(param);
			double df = a * steplength * J.zDotProduct(dp);
			if (fnew < fold + df) {
				steplength *= b;
			} else {
				break;
			}
		}
		return steplength;
	}

	/*
	 * perform one iteration of MLE maximization by BHHH method
	 */
	private boolean OneIterate(DoubleMatrix1D param) {
		DoubleMatrix2D dl = CalcJacobianTimeSeries(param);

		// dp = (dl*dl')\dl*ones(N,1)
		DoubleMatrix2D Hinv = null;// Inverse of Hessian Matrix
		try {
			Hinv = Alg.inverse(dl.zMult(dl.viewDice(), null));
		} catch (Exception ex) {
			Hinv = FacM.diagonal(FacV.make(5, 1.0));
		}
		DoubleMatrix1D J = dl.zMult(FacV.make(T, 1), null);
		DoubleMatrix1D dp = Hinv.zMult(J, null); // H^-1 * J

		double steplength = BacktrackingLineSearch(param, dp, J);

		//System.out.println("\t Step Length = " + steplength);
		param.assign(dp, Fn.plusMult(steplength));

		//System.out.println("\t Log_Likelihood = " + LogLikelihood(param));
		return J.zDotProduct(dp) < 1e-8 || steplength < 1e-8;
	}

	/*
	 * BHHH routine
	 */
	public double[] Calc_MLE_by_BHHH() {
		DoubleMatrix1D param = GenInitParamGuess();
		for (int i = 1; i < 50; i++) {
			//System.out.println("iter = " + i);
			if (OneIterate(param))
				break;
		}
		return param.toArray();
	}

	protected MaximisationFunction GenObjective() {
		return new MaximisationFunction() {
			public double function(double[] param) {
				return LogLikelihood(FacV.make(param));
			}
		};
	}

	/*
	 * using central difference to evaluation the Jacobian for one parameter
	 */
	private DoubleMatrix1D CalcOneParamJacobianTimeSeries(DoubleMatrix1D param, int i) {
		DoubleMatrix1D pp = param.copy();
		double z = param.get(i);
		double dz = Step[i];
		pp.set(i, z - dz);
		DoubleMatrix1D lm = CalcLogLikelihoodTimeSeries(pp);
		pp.set(i, z + dz);
		DoubleMatrix1D lp = CalcLogLikelihoodTimeSeries(pp);
		return lp.assign(lm, Fn.minus).assign(Fn.div(2 * dz));
	}

	/*
	 * calculate the Jacobian vector
	 */
	private DoubleMatrix2D CalcJacobianTimeSeries(DoubleMatrix1D param) {
		DoubleMatrix2D dl = FacM.make(N, T);
		for (int i = 0; i < N; i++) {
			DoubleMatrix1D deri = CalcOneParamJacobianTimeSeries(param, i);
			for (int j = 0; j < T; j++) {
				dl.set(i, j, deri.get(j));
			}
		}
		return dl;
	}

	/*
	 * calculate the Log Likelihood, which is the objective function to be
	 * maximized
	 */
	private double LogLikelihood(DoubleMatrix1D param) {
		double result = CalcLogLikelihoodTimeSeries(param).zSum();
		return (Double.isNaN(result) || Double.isInfinite(result)) ? 0 : result;
	}

	abstract protected DoubleMatrix1D GenInitParamGuess();

	/*
	 * calculate the MLE Nelder Mead Simplex method, this routine is to
	 * demostrate the correctness of the implementation of BHHH method BHHH
	 * methods is a Gauss-Newton Method, which is much faster in this
	 * application
	 */
	abstract public double[] Calc_MLE_by_NelderMead();

	abstract protected DoubleMatrix1D CalcLogLikelihoodTimeSeries(DoubleMatrix1D param);
}

/*
 * GarchRegress abstract class implementation for Normal Distribution
 */
class GarchRegress_Normal extends GarchRegress {
	GarchRegress_Normal(double[] x, double[] y) {
		super(x, y);
		this.N = 5;// # of parameters to be estimated
	}

	protected DoubleMatrix1D CalcLogLikelihoodTimeSeries(DoubleMatrix1D param) {
		DoubleMatrix1D[] series = Calc_e_and_h(param);
		DoubleMatrix1D e = series[0];
		DoubleMatrix1D h = series[1];
		return e.assign(Fn.square).assign(h, Fn.div).assign(h.assign(Fn.log), Fn.plus)
				.assign(Fn.mult(-0.5));
	}

	@Override
	public double[] Calc_MLE_by_NelderMead() {
		Maximisation max = new Maximisation();
		max.setNmax(5000);
		max.addConstraint(2, -1, 0.0);
		max.addConstraint(3, -1, 0.0);
		max.addConstraint(4, -1, 0.0);
		max.addConstraint(new int[] { 3, 4 }, new double[] { 1, 1 }, 1, 0.9);
		max.nelderMead(GenObjective(), GenInitParamGuess().toArray(), 1e-10);
		return max.getParamValues();
	}

	@Override
	protected DoubleMatrix1D GenInitParamGuess() {
		double[] p = { 0.1, 1.0, 0.2, 0.3, 0.4 };
		return FacV.make(p);
	}
}

/*
 * GarchRegress abstract class implementation for Student t Distribution
 */
class GarchRegress_StudT extends GarchRegress {
	GarchRegress_StudT(double[] x, double[] y) {
		super(x, y);
		this.N = 6;// # of parameters to be estimated, including deg. of freedom
	}

	protected DoubleMatrix1D CalcLogLikelihoodTimeSeries(DoubleMatrix1D param) {
		DoubleMatrix1D[] series = Calc_e_and_h(param);
		DoubleMatrix1D e = series[0];
		DoubleMatrix1D h = series[1];
		double df = param.get(5);

		double p1 = Gamma.logGamma((df + 1) / 2) - Gamma.logGamma(df / 2) - 0.5
				* Math.log((df - 2) * Math.PI);
		DoubleMatrix1D p2 = h.copy().assign(Fn.log).assign(Fn.mult(-0.5));
		DoubleMatrix1D p3 = e.assign(Fn.square).assign(h.assign(Fn.mult(df - 2)), Fn.div)
				.assign(Fn.plus(1)).assign(Fn.log).assign(Fn.mult(-(df + 1) * 0.5));
		return FacV.make(T, p1).assign(p2, Fn.plus).assign(p3, Fn.plus);
	}

	@Override
	public double[] Calc_MLE_by_NelderMead() {
		Maximisation max = new Maximisation();
		max.setNmax(5000);
		max.addConstraint(2, -1, 0.0);
		max.addConstraint(3, -1, 0.0);
		max.addConstraint(4, -1, 0.0);
		max.addConstraint(new int[] { 3, 4 }, new double[] { 1, 1 }, 1, 0.9);
		max.addConstraint(5, -1, 2.01);
		max.nelderMead(GenObjective(), GenInitParamGuess().toArray(), 1e-10);
		return max.getParamValues();
	}

	@Override
	protected DoubleMatrix1D GenInitParamGuess() {
		double[] p = { 0.1, 1.0, 0.2, 0.3, 0.4, 5.0 };
		return FacV.make(p);
	}
}

/*
 * GarchRegress abstract class implementation for Exponential Power Distribution
 */
class GarchRegress_ExpPow extends GarchRegress {
	GarchRegress_ExpPow(double[] x, double[] y) {
		super(x, y);
		this.N = 6;// # of parameters to be estimated, including tau
	}

	protected DoubleMatrix1D CalcLogLikelihoodTimeSeries(DoubleMatrix1D param) {
		double tau = param.get(5);
		DoubleMatrix1D[] series = Calc_e_and_h(param);
		DoubleMatrix1D e = series[0].assign(Fn.abs);
		DoubleMatrix1D s = series[1].assign(Fn.sqrt);
		double p1 = -(1 / tau + 1) * Math.log(2) - Gamma.logGamma(1 / tau + 1);
		DoubleMatrix1D p2 = e.assign(s, Fn.div).assign(Fn.pow(tau)).assign(Fn.mult(-0.5));
		return FacV.make(T, p1).assign(p2, Fn.plus).assign(s.assign(Fn.log), Fn.minus);
	}

	@Override
	public double[] Calc_MLE_by_NelderMead() {
		Maximisation max = new Maximisation();
		max.setNmax(5000);
		max.addConstraint(2, -1, 0.0);
		max.addConstraint(3, -1, 0.0);
		max.addConstraint(4, -1, 0.0);
		max.addConstraint(new int[] { 3, 4 }, new double[] { 1, 1 }, 1, 0.9);
		max.addConstraint(5, -1, 0.001);
		max.nelderMead(GenObjective(), GenInitParamGuess().toArray(), 1e-10);
		return max.getParamValues();
	}

	@Override
	protected DoubleMatrix1D GenInitParamGuess() {
		double[] p = { 0.1, 1.0, 0.2, 0.3, 0.4, 5.0 };
		return FacV.make(p);
	}
}
