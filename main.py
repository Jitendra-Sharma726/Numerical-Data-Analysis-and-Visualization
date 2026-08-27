import pandas as pd
import matplotlib.pyplot as plt

# --------- 1. Load Dataset ---------
def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv(file_path)
    return df

# --------- 2. Compute Statistical Values ---------
def compute_statistics(df: pd.DataFrame) -> dict:
    """
    Compute mean, median, mode, standard deviation, variance, and range
    for the 'Exam Score' column.
    Returns a dictionary of statistics.
    """
    exam_scores = df['Exam Score']
    
    stats = {
        "Mean": exam_scores.mean(),
        "Median": exam_scores.median(),
        "Mode": exam_scores.mode().tolist(),  # Converting to list for clean printing
        "Standard Deviation": exam_scores.std(),
        "Variance": exam_scores.var(),
        "Range": exam_scores.max() - exam_scores.min()
    }
    
    return stats

# --------- 3. Visualization ---------
def visualize_exam_score(df: pd.DataFrame):
    """
    Generate histogram and boxplot for 'Exam Score'.
    """
    exam_scores = df['Exam Score']
    
    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(exam_scores, bins=10, color='skyblue', edgecolor='black')
    plt.title('Distribution of Exam Scores')
    plt.xlabel('Exam Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('exam_score_histogram.png', bbox_inches='tight')
    plt.close()
    
    # Boxplot
    plt.figure(figsize=(6, 5))
    plt.boxplot(exam_scores, vert=True, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    plt.title('Box Plot of Exam Scores')
    plt.ylabel('Exam Score')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('exam_score_boxplot.png', bbox_inches='tight')
    plt.close()

# --------- 4. Outlier Detection using IQR ---------
def detect_outliers(df: pd.DataFrame) -> list:
    """
    Detect outliers in 'Exam Score' using IQR method.
    Returns a list of outlier values.
    """
    exam_scores = df['Exam Score']
    
    Q1 = exam_scores.quantile(0.25)
    Q3 = exam_scores.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = exam_scores[(exam_scores < lower_bound) | (exam_scores > upper_bound)].tolist()
    return outliers

# --------- Main Execution Block ---------
if __name__ == "__main__":
    file_path = "dataset.csv"
    
    # Load the dataset
    df = load_dataset(file_path)
    
    # Compute statistics
    stats = compute_statistics(df)
    
    # Print statistics
    for key, value in stats.items():
        print(f"{key}: {value}")
        
    # Generate visualizations
    visualize_exam_score(df)
    
    # Detect outliers
    outliers = detect_outliers(df)
    print(f"Outliers: {outliers}")
