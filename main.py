import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestIndPower

def get_user_input():
    # Get input from the user for the parameters
    while True:
        try:
            mean1 = float(input("Enter the mean ADAS-Cog change from baseline for the treatment group: "))
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    while True:
        try:
            stddev1 = float(input("Enter the standard deviation of the ADAS-Cog change for the treatment group: "))
            if stddev1 <= 0:
                print("Standard deviation must be positive.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    while True:
        try:
            dropout_rate = float(input("Enter the dropout rate (as a decimal between 0 and 1): "))
            if not 0 <= dropout_rate < 1:
                print("Dropout rate must be between 0 and 1.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a decimal value between 0 and 1.")

    n1 = math.ceil(402 * (1 - dropout_rate))  # Calculated sample size after dropout

    while True:
        try:
            mean2 = float(input("Enter the mean ADAS-Cog change from baseline for the placebo group: "))
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    while True:
        try:
            stddev2 = float(input("\nEnter the standard deviation of the ADAS-Cog change for the placebo group: "))
            if stddev2 <= 0:
                print("Standard deviation must be positive.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    n2 = n1  # Assuming equal sample sizes

    while True:
        enforce_max_delta = input("\nDo you want to enforce a maximum ADAS-Cog delta between the two groups? (y/n): ").lower()
        if enforce_max_delta in ['y', 'n']:
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    if enforce_max_delta == 'y':
        while True:
            try:
                max_delta = float(input("Enter the maximum allowed delta between group means: "))
                if max_delta <= 0:
                    print("Maximum delta must be positive.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
    else:
        max_delta = None

    return mean1, stddev1, n1, mean2, stddev2, n2, max_delta

def generate_samples(mean1, stddev1, n1, mean2, stddev2, n2, max_delta, do_not_print_details):
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
        print(f"Delta (Mean1 - Mean2): {delta:.4f}")
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
    # Perform a two-sample t-test (Welch's t-test)
    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=False)
    return t_stat, p_value

def calculate_effect_size(mean1, mean2, stddev1, stddev2, n1, n2):
    # Calculate Cohen's d for independent samples
    # Pooled standard deviation
    s_pooled = np.sqrt(((n1 - 1) * stddev1**2 + (n2 - 1) * stddev2**2) / (n1 + n2 - 2))
    effect_size = (mean1 - mean2) / s_pooled
    return effect_size

def calculate_power(effect_size, n1, n2, alpha=0.05):
    analysis = TTestIndPower()
    power = analysis.power(effect_size=effect_size, nobs1=n1, ratio=n2/n1, alpha=alpha, alternative='two-sided')
    return power

def main():
    # Get user inputs
    mean1, stddev1, n1, mean2, stddev2, n2, max_delta = get_user_input()
    do_not_print_details = False

    # Optional: Set seed once for reproducibility
    while True:
        use_seed = input("Do you want to set a seed for reproducibility? (y/n): ").lower()
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
    sample1, sample2 = generate_samples(mean1, stddev1, n1, mean2, stddev2, n2,
                                        max_delta, do_not_print_details)

    # Perform the t-test
    t_stat, p_value = perform_t_test(sample1, sample2)

    # Calculate effect size
    effect_size = calculate_effect_size(np.mean(sample1), np.mean(sample2), np.std(sample1, ddof=1), np.std(sample2, ddof=1), n1, n2)

    # Calculate power
    power = calculate_power(effect_size, n1, n2)

    # Output the results
    print(f"\nT-test Results:")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Effect Size (Cohen's d): {effect_size:.4f}")
    print(f"Power of the test: {power:.4f}")

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
            sample1, sample2 = generate_samples(mean1, stddev1, n1, mean2, stddev2, n2,
                                                max_delta, do_not_print_details)
            _, p_value_sim = perform_t_test(sample1, sample2)
            p_values.append(p_value_sim)

        # Calculate statistics for p-values
        avg_p_value = np.mean(p_values)
        max_p_value = np.max(p_values)
        min_p_value = np.min(p_values)
        stddev_p_value = np.std(p_values, ddof=1)

        # Output the results
        print(f"\nSimulation Results (n={simulation_size}):")
        print(f"Average p-value: {avg_p_value:.6f}")
        print(f"Maximum p-value: {max_p_value:.6f}")
        print(f"Minimum p-value: {min_p_value:.6f}")
        print(f"Standard deviation of p-values: {stddev_p_value:.6f}")

        # Optionally, plot the distribution of p-values
        while True:
            plot_hist = input("\nDo you want to see a histogram of the p-values? (y/n): ").lower()
            if plot_hist in ['y', 'n']:
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

        if plot_hist == 'y':
            plt.hist(p_values, bins=20, edgecolor='black', alpha=0.7)
            plt.title('Distribution of P-Values from Simulations')
            plt.xlabel('P-Value')
            plt.ylabel('Frequency')
            plt.show()
    else:
        print("Simulation skipped.")

if __name__ == "__main__":
    main()
