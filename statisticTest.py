import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

# מטרה רציפה -משתנה רציף-> spearman correlation
def paired_continuous_test(data: pd.DataFrame, column1: str, column2: str) -> tuple:
    """
    Perform spearman correlation test on two continuous columns.

    Parameters:
    - data (DataFrame): The DataFrame containing the data.
    - column1 (str): The name of the first continuous column.
    - column2 (str): The name of the second continuous column.

    Returns:
    - p_value (float): The p-value of the test.
    - corr (float): The correlation coefficient between the two columns.
    - fig: A matplotlib figure object showing the correlation of the two groups.
    """
    # prepare data
    data = data[[column1, column2]].dropna()
    groupA = data[column1]
    groupB = data[column2]

    # calculate spearman correalation
    corr, p_value = scipy.stats.spearmanr(groupA, groupB)
    # test for normality of the differences
    fig = plot_paired_distribution(groupA,groupB)
    plt.close(fig)
    return corr, p_value,fig

def plot_paired_distribution(groupA:np.ndarray,groupB:np.ndarray)->plt.Figure:
    """
    Plot the correlation of two paired groups.
    ### Parameters:
    - groupA (array-like): The data of the first group.
    - groupB (array-like): The data of the second group.
    ### Returns:
    - fig: A matplotlib figure object showing the relationship between the two groups.
    """
    fig = plt.figure(figsize=(5, 5))
    sns.regplot(x=groupA, y=groupB, ci=None)
    plt.title(f"Correlation between {groupA.name} and {groupB.name}")
    return fig

# מטרה לא רציפה -משתנה רציף-> mann whitney u test / unpaired t-test
# מטרה רציפה -משתנה בינארי-> mann whitney u test / unpaired t-test
def continuous_test(data: pd.DataFrame, continuous_column: str, binary_columns: str, binary_target = False) -> tuple:
    """
    Find whether to use unpaired t-test or Mann-Whitney U test on a continuous column based on a binary column.
    and plot the results.

    ### Parameters:
    - data (DataFrame): The DataFrame containing the data.
    - continuous_column (str): The name of the continuous column to be tested.
    - binary_columns (str): The name of the binary column to be split by.
    - binary_target (bool): Whether the binary column is the target variable (default: False). if True, the groups will be labeled as "Valid" and "InValid".

    ### Returns:
    - stat (float): The test statistic.<br>
    - p_value (float): The p-value of the test.
    - fig: A matplotlib figure object showing the distribution of the two groups.
    - is_normal (bool): Whether the data follows a normal distribution.
    - group_false_mean (float): The mean of the group with binary value 0 or the first unique value in the binary column.
    - group_true_mean (float): The mean of the group with binary value 1 or the second unique value in the binary column.
    """
    # prepare data
    data = data[[continuous_column, binary_columns]].dropna()
    groupFalse = data[data[binary_columns] == 0][continuous_column]
    groupTrue = data[data[binary_columns] == 1][continuous_column]
    group_false_mean = groupFalse.mean()
    group_true_mean = groupTrue.mean()
    if binary_target:
        groupFalse_title = "InValid"
        groupTrue_title = "Valid"
    else:
        groupFalse_title = f"No {binary_columns[::-1]}"
        groupTrue_title = f"Yes {binary_columns[::-1]}"
    # test for normality
    _, p_value = ks_normality_test(data[continuous_column])
    # decide which test to use
    if p_value > 0.05:
        # perform unpaired t-test
        t_stat, t_p_value = stats.ttest_ind(groupFalse, groupTrue,equal_var=True)

        # plot results
        fig = plot_distribution(groupFalse,groupTrue,groupFalse_title,groupTrue_title,continuous_column)
        plt.close(fig)
        return t_stat, t_p_value, fig ,group_false_mean,group_true_mean, True
    else:    
        # perform Mann-Whitney U test
        u_stat, u_p_value = stats.mannwhitneyu(groupFalse, groupTrue,alternative='two-sided')

        # plot results
        fig = plot_distribution(groupFalse,groupTrue,groupFalse_title,groupTrue_title,continuous_column)
        plt.close(fig)
        return u_stat, u_p_value,fig ,group_false_mean,group_true_mean, False
def plot_distribution(groupA:np.ndarray,groupB:np.ndarray,groupA_title,groupB_title,value="Value")->plt.Figure:
    """
    Plot the distribution of two independent groups.
    ### Parameters:
    - groupA (array-like): The data of group a.
    - groupB (array-like): The data of group b.
    - groupA_title (str): The title for group a.
    - groupB_title (str): The title for group b.
    - value (str): The name of the value being plotted (default: "Value").
    ### Returns:
    - fig: A matplotlib figure object showing the distribution of the two groups.
    """
    df = pd.DataFrame({
        value: np.concatenate([groupA, groupB]),
        "Group": [groupA_title] * len(groupA) + [groupB_title] * len(groupB)
    })

    fig = plt.figure(figsize=(6, 5))

    sns.boxplot(
        data=df,
        x="Group",
        y=value,
        palette=["#4C72B0", "#18CAAF"],
        width=0.5
    )
    plt.title(f"Distribution of {value} by Group")
    plt.xlabel("")
    plt.ylabel(value)
    return fig

def ks_normality_test(series: np.ndarray) -> tuple:
    """
    Perform the Kolmogorov-Smirnov test for normality on a given data column.

    ### Parameters:
    data (array-like): The data to be tested.
    
    ### Returns:
    - stat (float): The KS statistic.<br>
    - p_value (float): The p-value of the test.
    """

    mu = np.mean(series)
    sigma = np.std(series)
    
    # Perform KS test against a normal distribution
    stat, p_value = stats.kstest(series, 'norm', args=(mu, sigma))

    return stat, p_value

# מטרה לא רציפה-משתנה בינארי-> chi test
def binary_chi_test(data: pd.DataFrame, binary_column1: str, binary_column2: str) -> tuple:
    """
    Perform Chi-square test of independence between two binary columns in a DataFrame.

    ### Args:
    - data (DataFrame): The DataFrame containing the data.
    - binary_column1 (str): The name of the first binary column.
    - binary_column2 (str): The name of the second binary column.

    ### Returns:
    - chi2 (float): The Chi-square statistic.<br>
    - p_value (float): The p-value of the test.
    - fig: A matplotlib figure object showing the heatmap of the contingency table.
    """

    contingency_table = pd.crosstab(data[binary_column1], data[binary_column2])
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    # plot heatmap of contingency table
    fig = plt.figure(figsize=(5,5))
    sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Blues")
    plt.title("Contingency Table Heatmap")
    plt.close(fig)

    return chi2, p_value, fig

# distribution plots and data stats for continuous and binary features
def plot_pie(df, col):
    counts = df[col].value_counts(dropna=False)

    # Create readable labels
    labels = [
        "True" if label == 1 else
        "False" if label == 0 else
        "Missing"
        for label in counts.index
    ]

    # Plot
    fig = plt.figure(figsize=(6, 6))
    plt.pie(
        counts.values,
        labels=labels,
        startangle=90,
        autopct="%1.1f%%"
    )

    # Return counts as a dictionary
    result_counts = {label: int(count) for label, count in zip(labels, counts.values)}

    return fig, result_counts

def plot_continuous_distribution(df, col, bins=20):
    """
    Creates a histogram + KDE plot for continuous data.
    Returns the figure and basic statistics.
    """
    series = df[col].dropna()
    _, p = ks_normality_test(series)
    normal = p > 0.05

    # Basic stats
    stats = {
        "count": len(series),
        "mean": f"{float(series.mean()):.4f}",
        "median": f"{float(series.median()):.4f}",
        "std": f"{float(series.std()):.4f}",
        "min": f"{float(series.min()):.4f}",
        "max": f"{float(series.max()):.4f}",
        "is normal": normal
    }

    # Create figure
    fig = plt.figure(figsize=(5, 4))
    plt.hist(series, bins=bins, color="#1F77B4", alpha=0.7, edgecolor="black")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    return fig, stats