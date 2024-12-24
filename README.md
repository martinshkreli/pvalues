# Clinical Trial Script (Under Construction)

This Python script performs a statistical simulation and analysis using traditional hypothesis testing and the betting test methodology proposed by Glenn Shafer. It allows users to generate samples for treatment and placebo groups based on specified parameters, perform t-tests, calculate effect sizes, conduct power analysis, and compute betting scores to evaluate evidence against the null hypothesis.

## Features

- **Sample Generation**: Generates random samples for treatment and placebo groups using normal, t-distribution, or skewed normal distributions.
- **Dropout Rates**: Accounts for dropout rates in both groups.
- **Statistical Testing**: Performs two-sample t-tests to compare group means.
- **Effect Size Calculation**: Calculates Cohen's d and Hedges' g effect sizes.
- **Power Analysis**: Computes the statistical power of the test.
- **Betting Test**: Implements the betting test to evaluate evidence against the null hypothesis.
- **Simulations**: Optionally runs multiple simulations to observe p-value variability.
- **Visualizations**: Plots sample distributions and p-value histograms.

## Requirements

- Python 3.x
- Required Python packages:
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `statsmodels`

## Usage

### Running the Script

1. **Clone or Download** the script to your local machine.
2. **Ensure all dependencies are installed**. You can install the required packages using `pip`:

   ```bash
   pip install numpy matplotlib scipy statsmodels
   ```

3. **Run the script** from the command line:

   ```bash
   python betting_test_simulation.py
   ```

### User Inputs

When you run the script, it will prompt you to enter several parameters:

1. **Treatment Group Parameters**:
   - **Mean Change from Baseline**: The average change expected in the treatment group.
   - **Standard Deviation of Change**: Variability in the change for the treatment group.
   - **Dropout Rate**: Proportion of participants expected to drop out (0 to 1).
   - **Initial Sample Size**: Number of participants allocated to the treatment group before dropout.

2. **Placebo Group Parameters**:
   - **Mean Change from Baseline**: The average change expected in the placebo group.
   - **Standard Deviation of Change**: Variability in the change for the placebo group.
   - **Dropout Rate**: Proportion of participants expected to drop out (0 to 1).
   - **Initial Sample Size**: Number of participants allocated to the placebo group before dropout.

3. **Maximum Delta (Optional)**:
   - Decide whether to enforce a maximum allowed difference between group means during sample generation.

4. **Betting Test Parameters**:
   - **Alternative Hypothesis Mean**: Expected mean under the alternative hypothesis (often the treatment group mean).
   - **Alternative Hypothesis Standard Deviation**: Expected standard deviation under the alternative hypothesis.

5. **Distribution Choice**:
   - Choose the distribution to use for sample generation: `normal`, `t`, or `skewed_normal`.
   - If `t` is selected, provide degrees of freedom (>2).
   - If `skewed_normal` is selected, provide skewness parameters for both groups.

6. **Random Seed (Optional)**:
   - Decide whether to set a seed for reproducibility.
   - Enter an integer seed value if required.

7. **Visualization Options**:
   - Choose whether to view histograms of the sample distributions.
   - Decide if you want to see the distributions under the null and alternative hypotheses, and whether to save the plots.

8. **Simulations (Optional)**:
   - Decide whether to run simulations to observe p-value variability.
   - Enter the number of simulations to run.
   - Choose whether to view a histogram of the p-values from simulations.

## Outputs

After entering the inputs, the script will perform the analysis and display the results.

### Descriptive Statistics

Provides summary statistics for the generated samples:

- Delta between group means
- Means and standard deviations for both groups
- Skewness and kurtosis
- Minimum and maximum values

### T-test Results

Displays the results of the two-sample t-test:

- t-statistic
- Degrees of freedom used in the test
- p-value
- Cohen's d effect size
- Hedges' g effect size
- 95% confidence interval for the effect size
- Power of the test

### Betting Test Results

Shows the betting test outcomes:

- Total betting score (product of individual betting scores)
- Implied target (expected betting score under the alternative hypothesis)
- Interpretation of whether evidence against the null hypothesis is stronger than expected under the alternative

### Visualizations

- **Sample Distributions**: Histograms of the treatment and placebo group samples.
- **Hypothesis Distributions**: Plots of the null and alternative hypothesis distributions overlaid with the observed data.
- **P-value Histogram**: A histogram of p-values from simulations if this option is selected.

## Interpreting Results

- **Statistical Significance**: If the p-value is less than the chosen significance level (e.g., 0.05), the difference between group means is statistically significant.
- **Effect Size**: Cohen's d and Hedges' g indicate the magnitude of the difference. Values around 0.2, 0.5, and 0.8 typically represent small, medium, and large effects, respectively.
- **Power**: A power close to 1 indicates a high probability of correctly rejecting a false null hypothesis.
- **Betting Test**: A total betting score significantly higher than the implied target suggests strong evidence against the null hypothesis.
- **P-value Variability**: The distribution of p-values from simulations helps assess the consistency and reliability of the statistical test.

## Examples

## Dependencies

Ensure the following Python packages are installed:

- **NumPy**: `pip install numpy`
- **Matplotlib**: `pip install matplotlib`
- **SciPy**: `pip install scipy`
- **Statsmodels**: `pip install statsmodels`

Alternatively, install all dependencies at once:

```bash
pip install numpy matplotlib scipy statsmodels
```

## Notes

- **Random Seed**: Setting a random seed ensures reproducibility of the results.
- **Data Interpretation**: Statistical significance does not always imply practical significance; consider effect sizes and confidence intervals.
- **Betting Test**: Provides an alternative perspective on evidence against the null hypothesis; interpret in conjunction with traditional statistical tests.
- **Simulations**: Running simulations can be computationally intensive for large numbers; adjust according to your system's capabilities.
