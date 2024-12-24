import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from scipy.stats import skewnorm
from typing import Tuple, Optional  # For type hints

class BettingTest:
    """
    Class to perform the betting test as proposed by Glenn Shafer.
    Calculates betting scores and implied targets based on the null and alternative hypotheses.
    """

    def __init__(self, null_mean: float, null_std: float, alt_mean: float, alt_std: float):
        """
        Initialize the BettingTest with parameters for the null and alternative hypotheses.

        Parameters:
        - null_mean: Mean under the null hypothesis.
        - null_std: Standard deviation under the null hypothesis.
        - alt_mean: Mean under the alternative hypothesis.
        - alt_std: Standard deviation under the alternative hypothesis.
        """
        self.null_mean = null_mean
        self.null_std = null_std
        self.alt_mean = alt_mean
        self.alt_std = alt_std

    def bet(self, data: np.ndarray, distribution_choice: str,
            df: Optional[float] = None,
            skewness_null: Optional[float] = None,
            skewness_alt: Optional[float] = None) -> np.ndarray:
        """
        Calculate betting scores S(y) = q(y)/p(y) for an array of data points.

        Parameters:
        - data: Array of observed data points.
        - distribution_choice: The distribution used ('normal', 't', 'skewed_normal').
        - df: Degrees of freedom for t-distribution (if applicable).
        - skewness_null: Skewness parameter for null hypothesis (if applicable).
        - skewness_alt: Skewness parameter for alternative hypothesis (if applicable).

        Returns:
        - scores: Array of betting scores for each data point.
        """
        if distribution_choice == 'normal':
            p = stats.norm.pdf(data, loc=self.null_mean, scale=self.null_std)
            q = stats.norm.pdf(data, loc=self.alt_mean, scale=self.alt_std)
        elif distribution_choice == 't':
            if df is None:
                raise ValueError("Degrees of freedom (df) must be provided for t-distribution.")
            p = stats.t.pdf(data, df=df, loc=self.null_mean, scale=self.null_std)
            q = stats.t.pdf(data, df=df, loc=self.alt_mean, scale=self.alt_std)
        elif distribution_choice == 'skewed_normal':
            if skewness_null is None or skewness_alt is None:
                raise ValueError("Skewness parameters must be provided for skewed normal distribution.")
            p = skewnorm.pdf(data, a=skewness_null, loc=self.null_mean, scale=self.null_std)
            q = skewnorm.pdf(data, a=skewness_alt, loc=self.alt_mean, scale=self.alt_std)
        else:
            raise ValueError("Invalid distribution choice.")

        # Avoid division by zero and handle invalid values
        epsilon = 1e-10
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = q / (p + epsilon)
            scores = np.where(np.isfinite(scores), scores, 0)

        return scores

    def calculate_total_score(self, scores: np.ndarray) -> float:
        """
        Calculate the total betting score by multiplying individual scores.
        Uses logarithms to prevent numerical underflow.

        Parameters:
        - scores: Array of individual betting scores.

        Returns:
        - total_score: Total betting score (product of individual scores).
        """
        # Use log transformation to prevent numerical underflow
        # Ensure that scores are positive and avoid log(0)
        epsilon = 1e-10
        log_scores = np.log(np.maximum(scores, epsilon))
        total_log_score = np.sum(log_scores)
        total_score = np.exp(total_log_score)
        return total_score

    def implied_target(self) -> float:
        """
        Calculate the implied target S* = exp(E_Q[ln(S)]).
        This is the expected betting score under the alternative hypothesis.

        This calculation correctly accounts for different variances between the null and alternative distributions.

        Returns:
        - implied_target: The implied target score.
        """
        # Parameters
        mean_diff = self.alt_mean - self.null_mean          # μ_Q - μ_P
        variance_null = self.null_std ** 2                  # σ_P^2
        variance_alt = self.alt_std ** 2                    # σ_Q^2

        # Compute the KL divergence D(Q||P)
        log_scale_ratio = np.log(self.null_std / self.alt_std)  # ln(σ_P / σ_Q)
        variance_ratio_term = (variance_alt + mean_diff ** 2) / (2 * variance_null)  # [σ_Q^2 + (μ_Q - μ_P)^2 ] / (2σ_P^2)
        adjustment = -0.5  # Subtract 1/2 as per the KL divergence formula

        D_QP = log_scale_ratio + variance_ratio_term + adjustment
        implied_target = np.exp(D_QP)

        return implied_target

    def plot_distributions(self, data: np.ndarray, distribution_choice: str,
                           df: Optional[float] = None,
                           skewness_null: Optional[float] = None,
                           skewness_alt: Optional[float] = None,
                           save_plot: bool = False):
        """
        Plot the distributions under the null and alternative hypotheses, overlaying histograms of the data.

        Parameters:
        - data: Combined data from both groups.
        - distribution_choice: The distribution used ('normal', 't', 'skewed_normal').
        - df: Degrees of freedom for t-distribution (if applicable).
        - skewness_null: Skewness parameter for null hypothesis (if applicable).
        - skewness_alt: Skewness parameter for alternative hypothesis (if applicable).
        - save_plot: If True, saves the plot to a file.
        """
        x_min, x_max = np.min(data), np.max(data)
        x_values = np.linspace(x_min, x_max, 1000)
        if distribution_choice == 'normal':
            p_values = stats.norm.pdf(x_values, loc=self.null_mean, scale=self.null_std)
            q_values = stats.norm.pdf(x_values, loc=self.alt_mean, scale=self.alt_std)
        elif distribution_choice == 't':
            if df is None:
                raise ValueError("Degrees of freedom (df) must be provided for t-distribution.")
            p_values = stats.t.pdf(x_values, df=df, loc=self.null_mean, scale=self.null_std)
            q_values = stats.t.pdf(x_values, df=df, loc=self.alt_mean, scale=self.alt_std)
        elif distribution_choice == 'skewed_normal':
            if skewness_null is None or skewness_alt is None:
                raise ValueError("Skewness parameters must be provided for skewed normal distribution.")
            p_values = skewnorm.pdf(x_values, a=skewness_null, loc=self.null_mean, scale=self.null_std)
            q_values = skewnorm.pdf(x_values, a=skewness_alt, loc=self.alt_mean, scale=self.alt_std)
        else:
            raise ValueError("Invalid distribution choice.")

        plt.figure(figsize=(10, 6))
        plt.plot(x_values, p_values, label='Null Hypothesis Distribution (P)', color='blue')
        plt.plot(x_values, q_values, label='Alternative Hypothesis Distribution (Q)', color='orange')
        # Overlay histograms
        plt.hist(data, bins=30, density=True, alpha=0.5, color='grey', label='Observed Data')
        plt.title('Null and Alternative Hypothesis Distributions')
        plt.xlabel('Data Values')
        plt.ylabel('Probability Density')
        plt.legend()
        if save_plot:
            plt.savefig('hypothesis_distributions.png')
            print("Plot saved as 'hypothesis_distributions.png'.")
        plt.show()

def get_float(prompt: str) -> float:
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Invalid input. Please enter a numeric value (e.g., 1.23).")

def get_positive_float(prompt: str) -> float:
    while True:
        try:
            value = float(input(prompt))
            if value <= 0:
                print("Invalid input. Value must be positive.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a positive numeric value (e.g., 1.23).")

def get_positive_int(prompt: str) -> int:
    while True:
        try:
            value = int(input(prompt))
            if value <= 0:
                print("Invalid input. Value must be a positive integer.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter an integer value (e.g., 42).")

def get_decimal_between_0_and_1(prompt: str) -> float:
    while True:
        try:
            value = float(input(prompt))
            if not 0 <= value < 1:
                print("Invalid input. Value must be between 0 and 1 (e.g., 0.25).")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a decimal value between 0 and 1 (e.g., 0.25).")

def get_distribution(prompt: str) -> str:
    distributions = ['normal', 't', 'skewed_normal']
    print(f"Available distributions: {', '.join(distributions)}")
    while True:
        choice = input(prompt).lower()
        if choice in distributions:
            return choice
        else:
            print(f"Invalid input. Please choose from {', '.join(distributions)}.")

def get_user_input() -> Tuple[float, float, int, float, float, int, Optional[float],
                              float, float, str, Optional[float],
                              Optional[float], Optional[float]]:
    """
    Collect input from the user for the parameters needed for the simulation.
    Ensures that all inputs are valid and handles exceptions appropriately.

    Returns:
    - A tuple containing all the collected input parameters.
    """
    print("Please provide the following parameters for the treatment group:")
    mean1 = get_float("Enter the mean change from baseline for the treatment group: ")
    stddev1 = get_positive_float("Enter the standard deviation of the change for the treatment group: ")
    dropout_rate1 = get_decimal_between_0_and_1("Enter the dropout rate for the treatment group (as a decimal between 0 and 1): ")
    n1_initial = get_positive_int("Enter the initial sample size for the treatment group: ")
    n1 = max(1, math.ceil(n1_initial * (1 - dropout_rate1)))  # Ensure at least 1 sample

    print("\nPlease provide the following parameters for the placebo group:")
    mean2 = get_float("Enter the mean change from baseline for the placebo group: ")
    stddev2 = get_positive_float("Enter the standard deviation of the change for the placebo group: ")
    dropout_rate2 = get_decimal_between_0_and_1("Enter the dropout rate for the placebo group (as a decimal between 0 and 1): ")
    n2_initial = get_positive_int("Enter the initial sample size for the placebo group: ")
    n2 = max(1, math.ceil(n2_initial * (1 - dropout_rate2)))  # Ensure at least 1 sample

    # Enforce maximum delta between group means if desired
    while True:
        enforce_max_delta = input("\nDo you want to enforce a maximum delta between the two groups? (y/n): ").lower()
        if enforce_max_delta in ['y', 'n']:
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    if enforce_max_delta == 'y':
        max_delta = get_positive_float("Enter the maximum allowed delta between group means: ")
    else:
        max_delta = None

    # Collect input for the alternative hypothesis parameters used in the betting test
    print("\nFor the betting test, please provide parameters for the alternative hypothesis.")
    alt_mean = get_float("Enter the alternative hypothesis mean (e.g., treatment group mean): ")
    alt_stddev = get_positive_float("Enter the alternative hypothesis standard deviation: ")

    print("\nPlease select the distribution for generating samples:")
    distribution_choice = get_distribution("Enter the distribution name (normal/t/skewed_normal): ")

    # For t-distribution, collect degrees of freedom
    if distribution_choice == 't':
        while True:
            df = get_positive_float("Enter the degrees of freedom for the t-distribution (must be > 2): ")
            if df > 2:
                break
            else:
                print("Invalid input. Degrees of freedom must be greater than 2.")
    else:
        df = None

    # For skewed normal, collect skewness parameters for both groups
    if distribution_choice == 'skewed_normal':
        print("\nSkewness parameters for the skewed normal distribution:")
        skewness_treatment = get_float("Enter the skewness parameter for the treatment group (negative for left-skewed, positive for right-skewed): ")
        skewness_placebo = get_float("Enter the skewness parameter for the placebo group (negative for left-skewed, positive for right-skewed): ")
    else:
        skewness_treatment = None
        skewness_placebo = None

    return (mean1, stddev1, n1, mean2, stddev2, n2, max_delta, alt_mean, alt_stddev,
            distribution_choice, df, skewness_treatment, skewness_placebo)

def generate_samples(mean1: float, stddev1: float, n1: int,
                     mean2: float, stddev2: float, n2: int,
                     max_delta: Optional[float], do_not_print_details: bool,
                     distribution_choice: str, df: Optional[float],
                     skewness_treatment: Optional[float], skewness_placebo: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random samples for the treatment and placebo groups.
    Ensures that the difference between group means does not exceed the specified maximum delta.

    Parameters:
    - mean1, stddev1, n1: Parameters for the treatment group.
    - mean2, stddev2, n2: Parameters for the placebo group.
    - max_delta: Maximum allowed difference between group means.
    - do_not_print_details: If True, suppresses detailed print statements.
    - distribution_choice: The distribution to use ('normal', 't', 'skewed_normal').
    - df: Degrees of freedom for the t-distribution (if applicable).
    - skewness_treatment: Skewness parameter for the treatment group (if applicable).
    - skewness_placebo: Skewness parameter for the placebo group (if applicable).

    Returns:
    - sample1: Array of samples for the treatment group.
    - sample2: Array of samples for the placebo group.
    """
    max_attempts = 10000  # To prevent infinite loops
    attempts = 0
    while True:
        attempts += 1
        if attempts > max_attempts:
            raise Exception("Unable to generate samples satisfying the maximum delta constraint. Please adjust the parameters.")

        # Generate samples based on the chosen distribution
        if distribution_choice == 'normal':
            sample1 = np.random.normal(mean1, stddev1, n1)
            sample2 = np.random.normal(mean2, stddev2, n2)
        elif distribution_choice == 't':
            if df is None:
                raise ValueError("Degrees of freedom (df) must be provided for t-distribution.")
            sample1 = mean1 + stddev1 * stats.t.rvs(df=df, size=n1)
            sample2 = mean2 + stddev2 * stats.t.rvs(df=df, size=n2)
        elif distribution_choice == 'skewed_normal':
            if skewness_treatment is None or skewness_placebo is None:
                raise ValueError("Skewness parameters must be provided for skewed normal distribution.")
            sample1 = skewnorm.rvs(a=skewness_treatment, loc=mean1, scale=stddev1, size=n1)
            sample2 = skewnorm.rvs(a=skewness_placebo, loc=mean2, scale=stddev2, size=n2)
        else:
            raise ValueError("Invalid distribution choice.")

        if max_delta is not None:
            delta = abs(np.mean(sample1) - np.mean(sample2))
            if delta > max_delta:
                if not do_not_print_details:
                    print("Delta is too high, rerolling samples...")
                continue
        break  # Exit the loop if samples are acceptable

    if not do_not_print_details:
        # Print descriptive statistics
        delta = np.mean(sample1) - np.mean(sample2)
        print(f"\nDescriptive Statistics:")
        print(f"Delta (Treatment Mean - Placebo Mean): {delta:.4f}")
        print(f"Mean of Treatment Group: {np.mean(sample1):.4f}")
        print(f"Mean of Placebo Group: {np.mean(sample2):.4f}")
        print(f"Standard Deviation of Treatment Group: {np.std(sample1, ddof=1):.4f}")
        print(f"Standard Deviation of Placebo Group: {np.std(sample2, ddof=1):.4f}")
        print(f"Skewness of Treatment Group: {stats.skew(sample1):.4f}")
        print(f"Skewness of Placebo Group: {stats.skew(sample2):.4f}")
        print(f"Kurtosis of Treatment Group: {stats.kurtosis(sample1):.4f}")
        print(f"Kurtosis of Placebo Group: {stats.kurtosis(sample2):.4f}")
        print(f"Minimum of Treatment Group: {np.min(sample1):.4f}")
        print(f"Maximum of Treatment Group: {np.max(sample1):.4f}")
        print(f"Minimum of Placebo Group: {np.min(sample2):.4f}")
        print(f"Maximum of Placebo Group: {np.max(sample2):.4f}")

        # Option to plot the distributions
        while True:
            plot_distributions = input("\nDo you want to see histograms of the sample distributions? (y/n): ").lower()
            if plot_distributions in ['y', 'n']:
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

        if plot_distributions == 'y':
            plt.hist(sample1, bins=20, alpha=0.7, label='Treatment Group', density=True)
            plt.hist(sample2, bins=20, alpha=0.7, label='Placebo Group', density=True)
            plt.title('Sample Distributions')
            plt.xlabel('Change from Baseline')
            plt.ylabel('Density')
            plt.legend()
            plt.show()

    return sample1, sample2

def perform_t_test(sample1: np.ndarray, sample2: np.ndarray) -> Tuple[float, float, float]:
    """
    Perform a two-sample t-test on the provided samples.

    Parameters:
    - sample1: Array of samples for the treatment group.
    - sample2: Array of samples for the placebo group.

    Returns:
    - t_stat: The calculated t-statistic.
    - p_value: The p-value corresponding to the t-statistic.
    - df: Degrees of freedom used in the test.
    """
    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=False)
    # Degrees of freedom for Welch's t-test
    n1 = len(sample1)
    n2 = len(sample2)
    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)
    numerator = (var1 / n1 + var2 / n2) ** 2
    denominator = ((var1 / n1) ** 2) / (n1 - 1) + ((var2 / n2) ** 2) / (n2 - 1)
    df = numerator / denominator
    return t_stat, p_value, df

def calculate_effect_size(mean1: float, mean2: float,
                          stddev1: float, stddev2: float,
                          n1: int, n2: int) -> float:
    """
    Calculate Cohen's d effect size for independent samples.

    Parameters:
    - mean1, mean2: Means of the two groups.
    - stddev1, stddev2: Standard deviations of the two groups.
    - n1, n2: Sample sizes of the two groups.

    Returns:
    - effect_size: The calculated effect size (Cohen's d).
    """
    # Pooled standard deviation
    s_pooled = np.sqrt(
        ((n1 - 1) * stddev1 ** 2 + (n2 - 1) * stddev2 ** 2) / (n1 + n2 - 2)
    )
    effect_size = (mean1 - mean2) / s_pooled
    return effect_size


def calculate_hedges_g(d: float, n1: int, n2: int) -> float:
    """
    Calculate Hedges' g effect size which corrects Cohen's d for small sample bias.

    Parameters:
    - d: Cohen's d effect size.
    - n1, n2: Sample sizes of the two groups.

    Returns:
    - hedges_g: The adjusted effect size.
    """
    df = n1 + n2 - 2
    correction_factor = 1 - (3 / (4 * df - 1))
    hedges_g = d * correction_factor
    return hedges_g


def calculate_power(effect_size: float, n1: int, n2: int,
                    alpha: float = 0.05, alternative: str = 'two-sided') -> float:
    """
    Calculate the statistical power of the test using the specified effect size and sample sizes.

    Parameters:
    - effect_size: The effect size (Cohen's d) of the test.
    - n1, n2: Sample sizes of the two groups.
    - alpha: Significance level (default is 0.05).
    - alternative: 'two-sided', 'larger', or 'smaller'.

    Returns:
    - power: The calculated power of the test.
    """
    analysis = TTestIndPower()
    power = analysis.power(effect_size=effect_size, nobs1=n1,
                           ratio=n2 / n1, alpha=alpha, alternative=alternative)
    return power


def effect_size_confidence_interval(d: float, n1: int, n2: int,
                                    alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate the confidence interval for Cohen's d effect size.

    Parameters:
    - d: Effect size (Cohen's d).
    - n1, n2: Sample sizes of the two groups.
    - alpha: Significance level (default is 0.05).

    Returns:
    - ci_lower, ci_upper: The lower and upper bounds of the confidence interval.
    """
    from scipy.stats import t
    df = n1 + n2 - 2
    crit_value = t.ppf(1 - alpha / 2, df)
    se = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    ci_lower = d - crit_value * se
    ci_upper = d + crit_value * se
    return ci_lower, ci_upper


def main():
    """
    Main function to execute the simulation, perform statistical tests,
    and output the results to the user.
    """
    # Get user inputs
    (mean1, stddev1, n1, mean2, stddev2, n2, max_delta, alt_mean, alt_stddev,
     distribution_choice, df, skewness_treatment, skewness_placebo) = get_user_input()
    do_not_print_details = False

    # Optional: Set seed once for reproducibility
    while True:
        use_seed = input("\nDo you want to set a seed for reproducibility? (y/n): ").lower()
        if use_seed in ['y', 'n']:
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    if use_seed == 'y':
        while True:
            try:
                seed_value = int(input("Enter an integer seed value: "))
                np.random.seed(seed_value)
                break
            except ValueError:
                print("Invalid input. Please enter an integer value.")
    else:
        # Ensure randomness if no seed is set
        np.random.seed(None)

    # Generate samples based on the user's inputs
    sample1, sample2 = generate_samples(
        mean1, stddev1, n1, mean2, stddev2, n2,
        max_delta, do_not_print_details,
        distribution_choice, df, skewness_treatment, skewness_placebo
    )

    # Perform the t-test
    t_stat, p_value, df_used = perform_t_test(sample1, sample2)

    # Calculate effect size
    cohen_d = calculate_effect_size(
        np.mean(sample1), np.mean(sample2),
        np.std(sample1, ddof=1), np.std(sample2, ddof=1),
        n1, n2
    )

    # Adjust for small sample sizes using Hedges' g
    hedges_g = calculate_hedges_g(cohen_d, n1, n2)

    # Calculate power
    power = calculate_power(cohen_d, n1, n2, alternative='two-sided')

    # Calculate confidence interval for effect size
    ci_lower, ci_upper = effect_size_confidence_interval(cohen_d, n1, n2)

    # Betting Test
    # Combine samples into a single data set for betting
    combined_data = np.concatenate((sample1, sample2))
    betting_test = BettingTest(
        null_mean=mean2, null_std=stddev2,
        alt_mean=alt_mean, alt_std=alt_stddev
    )

    # Calculate betting scores for combined data
    betting_scores = betting_test.bet(
        combined_data, distribution_choice, df,
        skewness_null=skewness_placebo, skewness_alt=skewness_treatment
    )
    total_betting_score = betting_test.calculate_total_score(betting_scores)
    implied_target = betting_test.implied_target()

    # Output the results
    print(f"\nT-test Results:")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"Degrees of freedom: {df_used:.2f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Effect Size (Cohen's d): {cohen_d:.4f}")
    print(f"Effect Size (Hedges' g): {hedges_g:.4f}")
    print(f"95% Confidence Interval for Effect Size: ({ci_lower:.4f}, {ci_upper:.4f})")
    print(f"Power of the test: {power:.4f}")

    # Output Betting Test Results
    print(f"\nBetting Test Results:")
    print(f"Total Betting Score: {total_betting_score:.4e}")
    print(f"Implied Target (Expected Score): {implied_target:.4f}")

    # Interpret the betting score
    if total_betting_score > implied_target:
        print("Evidence against the null hypothesis is stronger than expected under the alternative.")
    else:
        print("Evidence against the null hypothesis is not stronger than expected under the alternative.")

    # Plotting the distributions
    while True:
        plot_betting_distributions = input("\nDo you want to see the distributions under the null and alternative hypotheses? (y/n): ").lower()
        if plot_betting_distributions in ['y', 'n']:
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    if plot_betting_distributions == 'y':
        save_plot_choice = input("Do you want to save the plot as an image file? (y/n): ").lower()
        save_plot = save_plot_choice == 'y'
        betting_test.plot_distributions(
            combined_data, distribution_choice, df,
            skewness_null=skewness_placebo, skewness_alt=skewness_treatment, save_plot=save_plot
        )

    # Ask if the user wants to run simulations
    while True:
        run_simulations = input("\nDo you want to run simulations to observe p-value variability? (y/n): ").lower()
        if run_simulations in ['y', 'n']:
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    if run_simulations == "y":
        while True:
            try:
                simulation_size = int(input("Enter the number of simulations to run: "))
                if simulation_size <= 0:
                    print("Invalid input. Please enter a positive integer.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter an integer value.")
        do_not_print_details = True
        p_values = []
        for _ in range(simulation_size):
            sample1_sim, sample2_sim = generate_samples(
                mean1, stddev1, n1, mean2, stddev2, n2,
                max_delta, do_not_print_details,
                distribution_choice, df, skewness_treatment, skewness_placebo
            )
            _, p_value_sim, _ = perform_t_test(sample1_sim, sample2_sim)
            p_values.append(p_value_sim)

        # Calculate statistics for p-values
        avg_p_value = np.mean(p_values)
        max_p_value = np.max(p_values)
        min_p_value = np.min(p_values)
        stddev_p_value = np.std(p_values, ddof=1)

        # Output the p-value results
        print(f"\nP-Value Simulation Results (n={simulation_size}):")
        print(f"Average p-value: {avg_p_value:.6f}")
        print(f"Maximum p-value: {max_p_value:.6f}")
        print(f"Minimum p-value: {min_p_value:.6f}")
        print(f"Standard Deviation of p-values: {stddev_p_value:.6f}")

        # Optionally, plot the distribution of p-values
        while True:
            plot_hist = input("\nDo you want to see a histogram of the p-values? (y/n): ").lower()
            if plot_hist in ['y', 'n']:
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

        if plot_hist == 'y':
            # Plot p-values
            plt.figure(figsize=(10, 6))
            plt.hist(p_values, bins=20, edgecolor='black', alpha=0.7)
            plt.title('Distribution of P-Values from Simulations')
            plt.xlabel('P-Value')
            plt.ylabel('Frequency')
            plt.show()
    else:
        print("Simulation skipped.")


if __name__ == "__main__":
    main()
