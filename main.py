import numpy as np
import math
from scipy import stats

def get_user_input():
    # Get input from the user for the parameters
    #ensure all inputs are valid types
    try:
        mean1 = float(input("Enter the mean ADAS-Cog change from baseline for simufilam: "))
    except ValueError:
        print("Invalid mean. Please enter a positive value.")
        mean1 = float(input("Enter the mean ADAS-Cog change from baseline for simufilam: "))
    while mean1 < 0:
        print("Invalid mean. Please enter a positive value.")
        mean1 = float(input("Enter the mean ADAS-Cog change from baseline for simufilam: "))

    try:
        stddev1 = float(input("Enter the standard deviation of the ADAS-Cog change from baseline for simufilam: "))
    except ValueError:
        print("Invalid standard deviation. Please enter a positive value.")
        stddev1 = float(input("Enter the standard deviation of the ADAS-Cog change from baseline for simufilam: "))
    while stddev1 < 0:
        print("Invalid standard deviation. Please enter a positive value.")
        stddev1 = float(input("Enter the standard deviation of the ADAS-Cog change from baseline for simufilam: "))
    try:
        dropout_rate = float(input("Enter the dropout rate (as a decimal): "))
    except ValueError:
        print("Invalid dropout rate. Please enter a positive value.")
        dropout_rate = float(input("Enter the dropout rate (as a decimal): "))
    while dropout_rate < 0 or dropout_rate > 1:
        print("Invalid dropout rate. Please enter a value between 0 and 1.")
        dropout_rate = float(input("Enter the dropout rate (as a decimal): "))
    #n1 = int(input("Enter the sample size of the first distribution: "))
    n1 = math.ceil(402 * (1 - dropout_rate))
    try:
        mean2 = float(input("Enter the mean ADAS-Cog change from baseline for placebo: "))
    except ValueError:
        print("Invalid mean. Please enter a positive value.")
        mean2 = float(input("Enter the mean ADAS-Cog change from baseline for placebo: "))
    while mean2 < 0:
        print("Invalid mean. Please enter a positive value.")
        mean2 = float(input("Enter the mean ADAS-Cog change from baseline for placebo: "))
    try:
        print("Standard deviation is one of the most important parameters. It should be a value that is consistent with the literature. Donanemab had a SD of 6-7 for drug and placebo.")
        stddev2 = float(input("Enter the standard deviation of the ADAS-Cog change from baseline for placebo: "))
    except ValueError:
        print("Invalid standard deviation. Please enter a positive value.")
        stddev2 = float(input("Enter the standard deviation of the ADAS-Cog change from baseline for placebo: "))
    while stddev2 < 0:
        print("Invalid standard deviation. Please enter a positive value.")
        stddev2 = float(input("Enter the standard deviation of the ADAS-Cog change from baseline for placebo: "))
    #n2 = int(input("Enter the sample size of the second distribution: "))
    #make sure n2 is an integer, round up if necessary
    n2 = math.ceil(402 * (1 - dropout_rate))
    use_same_seed = True
    seed1, seed2 = 42, 42
    use_same_seed = input("Do you want to use the exact same distribution for both groups or randomize? (y/n): ")
    #do you want to enforce a maximum delta
    max_delta = input("There will be some randomness when creating the distributions. Do you want to enforce a maximum ADAS-Cog delta between the two groups? (y (enforce)/n (let it roll)): ")
    if max_delta == "y":
        max_delta = float(input("Enter the maximum delta: "))
    else:
        max_delta = float('inf')
    if use_same_seed == "y":
        use_random_seed = False
    else:
        use_random_seed = True
    return mean1, stddev1, n1, mean2, stddev2, n2, seed1, seed2, use_random_seed, max_delta

def generate_samples(mean1, stddev1, n1, mean2, stddev2, n2, seed1, seed2, use_random_seed, do_not_print_details, max_delta):
    if use_random_seed:
        seed1, seed2 = np.random.randint(0, 1000000), np.random.randint(0, 1000000)
    else:
        seed1, seed2 = 42, 42

    np.random.seed(seed1)  # Any seed value works, it just needs to be the same
    sample1 = np.random.normal(mean1, stddev1, n1)    
    np.random.seed(seed2)  # Reset the seed to ensure identical sample
    sample2 = np.random.normal(mean2, stddev2, n2)
    #delta should be absolute value
    delta = abs(np.mean(sample1) - np.mean(sample2))
    while delta > max_delta:
        print("Delta is too high, rerolling samples")
        sample1, sample2 = generate_samples(mean1, stddev1, n1, mean2, stddev2, n2, seed1, seed2, use_random_seed, do_not_print_details, max_delta)
        delta = np.mean(sample1) - np.mean(sample2)

    if not do_not_print_details:
        #print the samples and provide more descriptive statistics
        print(f"Sample 1: {sample1}", end="\t")
        print(f"Sample 2: {sample2}")
        #delta
        delta = np.mean(sample1) - np.mean(sample2)
        print(f"Delta: {delta}")
        print(f"Mean of Sample 1: {np.mean(sample1)}", end="\t")
        print(f"Mean of Sample 2: {np.mean(sample2)}")
        print(f"Standard Deviation of Sample 1: {np.std(sample1)}", end="\t")
        print(f"Standard Deviation of Sample 2: {np.std(sample2)}")
        print(f"Variance of Sample 1: {np.var(sample1)}", end="\t")
        print(f"Variance of Sample 2: {np.var(sample2)}")
        print(f"Skewness of Sample 1: {stats.skew(sample1)}", end="\t")
        print(f"Skewness of Sample 2: {stats.skew(sample2)}")
        print(f"Kurtosis of Sample 1: {stats.kurtosis(sample1)}")
        print(f"Kurtosis of Sample 2: {stats.kurtosis(sample2)}")
        print(f"Minimum of Sample 1: {np.min(sample1)}", end="\t")
        print(f"Maximum of Sample 1: {np.max(sample1)}")
        print(f"Minimum of Sample 2: {np.min(sample2)}", end="\t")
        print(f"Maximum of Sample 2: {np.max(sample2)}")
    return sample1, sample2

def perform_t_test(sample1, sample2):
    # Perform a two-sample t-test
    t_stat, p_value = stats.ttest_ind(sample1, sample2)
    return t_stat, p_value

def main():
    # Get user inputs
    mean1, stddev1, n1, mean2, stddev2, n2, seed1, seed2, use_random_seed, max_delta = get_user_input()
    do_not_print_details = False
    # Generate samples based on the user's inputs
    sample1, sample2 = generate_samples(mean1, stddev1, n1, mean2, stddev2, n2, seed1, seed2, use_random_seed, do_not_print_details, max_delta)
    
    # Perform the t-test
    t_stat, p_value = perform_t_test(sample1, sample2)
    
    # Output the results
    print(f"t-statistic: {t_stat}")
    print(f"p-value: {p_value}")

    #ask for simulation size
    simulation_size = int(input("Enter the number of simulations to run: "))
    do_not_print_details = True
     # Simulate 1000 trials
    p_values = []
    for _ in range(simulation_size):
        sample1, sample2 = generate_samples(mean1, stddev1, n1, mean2, stddev2, n2, seed1, seed2, use_random_seed, do_not_print_details, max_delta)
        _, p_value = perform_t_test(sample1, sample2)
        p_values.append(p_value)
    
    # Calculate statistics for p-values
    avg_p_value = np.mean(p_values)
    max_p_value = np.max(p_values)
    min_p_value = np.min(p_values)
    stddev_p_value = np.std(p_values)
    
    # Output the results
    print(f"Average p-value: {avg_p_value}")
    print(f"Maximum p-value: {max_p_value}")
    print(f"Minimum p-value: {min_p_value}")
    print(f"Standard deviation of p-values: {stddev_p_value}")



if __name__ == "__main__":
    main()
