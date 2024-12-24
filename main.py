import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.power import TTestIndPower


class BettingTest:
    """
    Class to perform the betting test as proposed by Glenn Shafer.
    Calculates betting scores and implied targets based on the null and alternative hypotheses.
    """

    def __init__(self, null_mean, null_std, alt_mean, alt_std):
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

    def bet(self, data):
        """
        Calculate betting scores S(y) = q(y)/p(y) for an array of data points.

        Parameters:
        - data: Array of observed data points.

        Returns:
        - scores: Array of betting scores for each data point.
        """
        # Density under null hypothesis P
        p = stats.norm.pdf(data, loc=self.null_mean, scale=self.null_std)

        # Density under alternative hypothesis Q
        q = stats.norm.pdf(data, loc=self.alt_mean, scale=self.alt_std)

        # Avoid division by zero and handle invalid values
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = np.where(p != 0, q / p, 0)

        return scores

    def calculate_total_score(self, scores):
        """
        Calculate the total betting score by multiplying individual scores.
        Uses logarithms to prevent numerical underflow.

        Parameters:
        - scores: Array of individual betting scores.

        Returns:
        - total_score: Total betting score (product of individual scores).
        """
        # Use log transformation to prevent numerical underflow
        log_scores = np.log(scores)
        total_log_score = np.sum(log_scores)
        total_score = np.exp(total_log_score)
        return total_score

    def implied_target(self):
        """
        Calculate the implied target S* = exp(E_Q[ln(S)]).
        This is the expected betting score under the alternative hypothesis.

        Returns:
        - implied_target: The implied target score.
        """
        mean_diff = self.alt_mean - self.null_mean
        exponent = (mean_diff ** 2) / (2 * self.null_std ** 2)
        return np.exp(exponent)


def get_user_input():
    """
    Collect input from the user for the parameters needed for the simulation.
    Ensures that all inputs are valid and handles exceptions appropriately.

    Returns:
    - A tuple containing all the collected input parameters.
    """
    # Helper function to get a positive float value
    def get_positive_float(prompt):
        while True:
            try:
                value = float(input(prompt))
                if value <= 0:
                    print("Value must be positive.")
                    continue
                return value
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    # Helper function to get a float value between 0 and 1
    def get_decimal_between_0_and_1(prompt):
        while True:
            try:
                value = float(input(prompt))
                if not 0 <= value < 1:
                    print("Value must be between 0 and 1.")
                    continue
                return value
            except ValueError:
                print("Invalid input. Please enter a decimal value between 0 and 1.")

    print("Please provide the following parameters for the treatment group:")
    mean1 = get_positive_float("Enter the mean ADAS-Cog change from baseline for the treatment group: ")
    stddev1 = get_positive_float("Enter the standard deviation of the ADAS-Cog change for the treatment group: ")
    dropout_rate = get_decimal_between_0_and_1("Enter the dropout rate (as a decimal between 0 and 1): ")
    n1 = math.ceil(402 * (1 - dropout_rate))  # Calculated sample size after dropout

    print("\nPlease provide the following parameters for the placebo group:")
    mean2 = get_positive_float("Enter the mean ADAS-Cog change from baseline for the placebo group: ")
    stddev2 = get_positive_float("Enter the standard deviation of the ADAS-Cog change for the placebo group: ")
    n2 = n1  # Assuming equal sample sizes

    # Enforce maximum delta between group means if desired
    while True:
        enforce_max_delta = input("\nDo you want to enforce a maximum ADAS-Cog delta between the two groups? (y/n): ").lower()
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
    alt_mean = get_positive_float("Enter the alternative hypothesis mean (e.g., treatment group mean): ")
    alt_stddev = get_positive_float("Enter the alternative hypothesis standard deviation: ")

    return mean1, stddev1, n1, mean2, stddev2, n2, max_delta, alt_mean, alt_stddev


def generate_samples(mean1, stddev1, n1, mean2, stddev2, n2, max_delta, do_not_print_details):
    """
    Generate random samples for the treatment and placebo groups.
    Ensures that the difference between group means does not exceed the specified maximum delta.

    Parameters:
    - mean1, stddev1, n1: Parameters for the treatment group.
    - mean2, stddev2, n2: Parameters for the placebo group.
    - max_delta: Maximum allowed difference between group means.
    - do_not_print_details: If True, suppresses detailed print statements.

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

        # Generate random samples
        sample1 = np.random.normal(mean1, stddev1, n1)
        sample2 = np.random.normal(mean2, stddev2, n2)

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
            plt.hist(sample1, bins=20, alpha=0.7, label='Treatment Group')
            plt.hist(sample2, bins=20, alpha=0.7, label='Placebo Group')
            plt.title('Sample Distributions')
            plt.xlabel('ADAS-Cog Change')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()

    return sample1, sample2


def perform_t_test(sample1, sample2):
    """
    Perform a two-sample t-test (Welch's t-test) on the provided samples.

    Parameters:
    - sample1: Array of samples for the treatment group.
    - sample2: Array of samples for the placebo group.

    Returns:
    - t_stat: The calculated t-statistic.
    - p_value: The p-value corresponding to the t-statistic.
    """
    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=False)
    return t_stat, p_value


def calculate_effect_size(mean1, mean2, stddev1, stddev2, n1, n2):
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
    s_pooled = np.sqrt(((n1 - 1) * stddev1 ** 2 + (n2 - 1) * stddev2 ** 2) / (n1 + n2 - 2))
    effect_size = (mean1 - mean2) / s_pooled
    return effect_size


def calculate_power(effect_size, n1, n2, alpha=0.05):
    """
    Calculate the statistical power of the test using the specified effect size and sample sizes.

    Parameters:
    - effect_size: The effect size (Cohen's d) of the test.
    - n1, n2: Sample sizes of the two groups.
    - alpha: Significance level (default is 0.05).

    Returns:
    - power: The calculated power of the test.
    """
    analysis = TTestIndPower()
    power = analysis.power(effect_size=effect_size, nobs1=n1, ratio=n2 / n1, alpha=alpha, alternative='two-sided')
    return power


def main():
    """
    Main function to execute the simulation, perform statistical tests,
    and output the results to the user.
    """
    # Get user inputs
    mean1, stddev1, n1, mean2, stddev2, n2, max_delta, alt_mean, alt_stddev = get_user_input()
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
        max_delta, do_not_print_details
    )

    # Perform the t-test
    t_stat, p_value = perform_t_test(sample1, sample2)

    # Calculate effect size
    effect_size = calculate_effect_size(
        np.mean(sample1), np.mean(sample2),
        np.std(sample1, ddof=1), np.std(sample2, ddof=1),
        n1, n2
    )

    # Calculate power
    power = calculate_power(effect_size, n1, n2)

    # Betting Test
    # Combine samples into a single data set for betting
    combined_data = np.concatenate((sample1, sample2))
    betting_test = BettingTest(
        null_mean=mean2, null_std=stddev2,
        alt_mean=alt_mean, alt_std=alt_stddev
    )

    # Calculate betting scores for combined data
    betting_scores = betting_test.bet(combined_data)
    total_betting_score = betting_test.calculate_total_score(betting_scores)
    implied_target = betting_test.implied_target()

    # Output the results
    print(f"\nT-test Results:")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Effect Size (Cohen's d): {effect_size:.4f}")
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
                    print("Please enter a positive integer.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter an integer value.")
        do_not_print_details = True
        p_values = []
        for _ in range(simulation_size):
            sample1_sim, sample2_sim = generate_samples(
                mean1, stddev1, n1, mean2, stddev2, n2,
                max_delta, do_not_print_details
            )
            _, p_value_sim = perform_t_test(sample1_sim, sample2_sim)
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
            plt.hist(p_values, bins=20, edgecolor='black', alpha=0.7)
            plt.title('Distribution of P-Values from Simulations')
            plt.xlabel('P-Value')
            plt.ylabel('Frequency')
            plt.show()
    else:
        print("Simulation skipped.")


if __name__ == "__main__":
    main()
