# Clinical Trial Simulation and Statistical Analysis (Experimental #1)

This script simulates clinical trial data and performs statistical analyses using both traditional statistical methods and Glenn Shafer's betting framework.

## Features

- **Data Simulation**: Generate synthetic data for treatment and placebo groups based on user-defined parameters.
- **Statistical Testing**: Perform Welch's t-test to compute t-statistics and p-values.
- **Effect Size and Power Analysis**: Calculate Cohen's d effect size and the statistical power of the test.
- **Betting Test**: Implement Glenn Shafer's betting framework to evaluate evidence against the null hypothesis.
- **Simulations**: Run multiple simulations to observe variability in p-values.
- **Data Visualization**: Optionally display histograms of sample distributions and p-values.

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib
- Statsmodels

Install the required packages using:

```bash
pip install numpy scipy matplotlib statsmodels
```

## Usage

Run the script:

```bash
python main.py
```

**User Inputs:**

- Means and standard deviations for treatment and placebo groups.
- Dropout rate (decimal between 0 and 1).
- Option to enforce a maximum allowed difference (delta) between group means.
- Parameters for the alternative hypothesis (used in the betting test).
- Option to set a random seed for reproducibility.
- Option to run simulations and specify the number of simulations.

**Outputs:**

- Descriptive statistics of the generated samples.
- T-test results: t-statistic and p-value.
- Effect size (Cohen's d) and power of the test.
- Betting test results: total betting score and implied target.
- Interpretation of the betting score relative to the implied target.
- (Optional) Histograms of sample distributions and p-values.

## Interpretation

- **P-Value**: Probability of observing the data, or something more extreme, under the null hypothesis.
- **Effect Size (Cohen's d)**: Measures the magnitude of the difference between two groups.
- **Power**: Probability of correctly rejecting the null hypothesis when it is false.
- **Betting Score**: Represents the strength of evidence against the null hypothesis based on the betting framework.
- **Implied Target**: Expected betting score under the alternative hypothesis; used as a benchmark to interpret the betting score.

## Notes

- Setting a random seed ensures reproducibility of the simulated data.
- The betting framework provides an alternative perspective on statistical evidence, as proposed by Glenn Shafer.
- Ensure input parameters are realistic to obtain meaningful simulation results.
- The script uses logarithms in the betting score calculation to maintain numerical stability.
