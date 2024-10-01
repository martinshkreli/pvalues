import numpy as np
from scipy import stats

def get_user_input():
    # Get input from the user for the parameters
    mean1 = float(input("Enter the mean of the first distribution: "))
    stddev1 = float(input("Enter the standard deviation of the first distribution: "))
    n1 = int(input("Enter the sample size of the first distribution: "))
    
    mean2 = float(input("Enter the mean of the second distribution: "))
    stddev2 = float(input("Enter the standard deviation of the second distribution: "))
    n2 = int(input("Enter the sample size of the second distribution: "))
    
    return mean1, stddev1, n1, mean2, stddev2, n2

def generate_samples(mean1, stddev1, n1, mean2, stddev2, n2):
    # Use the same seed before generating both samples to ensure identical distributions
    np.random.seed(42)  # Any seed value works, it just needs to be the same
    sample1 = np.random.normal(mean1, stddev1, n1)
    
    np.random.seed(42)  # Reset the seed to ensure identical sample
    sample2 = np.random.normal(mean2, stddev2, n2)
    #print the samples and provide more descriptive statistics
    print(f"Sample 1: {sample1}")
    print(f"Sample 2: {sample2}")
    print(f"Mean of Sample 1: {np.mean(sample1)}")
    print(f"Mean of Sample 2: {np.mean(sample2)}")
    print(f"Standard Deviation of Sample 1: {np.std(sample1)}")
    print(f"Standard Deviation of Sample 2: {np.std(sample2)}")
    print(f"Variance of Sample 1: {np.var(sample1)}")
    print(f"Variance of Sample 2: {np.var(sample2)}")
    print(f"Skewness of Sample 1: {stats.skew(sample1)}")
    print(f"Skewness of Sample 2: {stats.skew(sample2)}")
    print(f"Kurtosis of Sample 1: {stats.kurtosis(sample1)}")
    print(f"Kurtosis of Sample 2: {stats.kurtosis(sample2)}")
    print(f"Minimum of Sample 1: {np.min(sample1)}")
    print(f"Maximum of Sample 1: {np.max(sample1)}")
    print(f"Minimum of Sample 2: {np.min(sample2)}")
    print(f"Maximum of Sample 2: {np.max(sample2)}")
    return sample1, sample2

def perform_t_test(sample1, sample2):
    # Perform a two-sample t-test
    t_stat, p_value = stats.ttest_ind(sample1, sample2)
    return t_stat, p_value

def main():
    # Get user inputs
    mean1, stddev1, n1, mean2, stddev2, n2 = get_user_input()
    
    # Generate samples based on the user's inputs
    sample1, sample2 = generate_samples(mean1, stddev1, n1, mean2, stddev2, n2)
    
    # Perform the t-test
    t_stat, p_value = perform_t_test(sample1, sample2)
    
    # Output the results
    print(f"t-statistic: {t_stat}")
    print(f"p-value: {p_value}")

    # Simulate 1000 trials
    p_values = []
    for _ in range(1000):
        sample1, sample2 = generate_samples(mean1, stddev1, n1, mean2, stddev2, n2)
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
