import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
from ..utils import printmd


def duplicated_report(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Generates a report on the number of duplicates in the specified column(s) of a pandas DataFrame.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        columns (str or list): The column name(s) to analyze for duplicates.
    """
    if isinstance(columns, str):
        columns = [columns]
    
    report = "### Duplicates Report\n\n"
    total_rows = len(df)
    
    for column in columns:
        duplicated_count = df.duplicated(subset=column).sum()
        duplicated_percentage = (duplicated_count / total_rows) * 100
        report += f"- **{column}**: {duplicated_count} duplicates ({duplicated_percentage:.2f}%)\n"
    
    printmd(report)

def missing_values_report(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Generates a report on the number of missing values (nulls) in the specified column(s) of a pandas DataFrame.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        columns (str or list): The column name(s) to analyze for missing values.
    """
    if isinstance(columns, str):
        columns = [columns]
    
    report = "### Missing Values Report\n\n"
    total_rows = len(df)
    
    for column in columns:
        missing_count = df[column].isnull().sum()
        missing_percentage = (missing_count / total_rows) * 100
        report += f"- **{column}**: {missing_count} missing values ({missing_percentage:.2f}%)\n"
    
    printmd(report)


def plot_distribution(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Plots the distribution of specified columns in the DataFrame.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing the columns to plot.
        columns (list): List of column names to plot the distribution.
    """
    # Set up the figure and axes for plotting
    fig, axes = plt.subplots(nrows=len(columns), figsize=(8, 4 * len(columns)))
    
    # Generate distribution plots for each column
    for i, col in enumerate(columns):
        ax = axes[i]
        sns.histplot(data=df[col], ax=ax, kde=True)
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {col}')
    
    # Adjust the layout and display the plots
    plt.tight_layout()
    plt.show()