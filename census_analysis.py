"""
Census 2022 sample data cleaning and exploratory analysis.

Usage:
    python census_analysis.py
    python census_analysis.py --input Census2022sample_F19.csv --output-dir output

Place the CSV file in the same directory as this script, or pass its path with --input.
The script writes cleaned data, summary reports, and charts to the output directory.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Suitable for deployment/server environments.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Clean and analyse Census 2022 sample data.")
    parser.add_argument(
        "--input",
        default="Census2022sample_F19.csv",
        help="Path to the input Census CSV file (default: Census2022sample_F19.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for cleaned data, reports, and charts (default: output)",
    )
    return parser.parse_args()


def save_plot(output_dir, filename):
    """Save the current plot and close it to keep memory use under control."""
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find input CSV: {input_path.resolve()}\n"
            "Put Census2022sample_F19.csv next to this script or use --input."
        )

    # Load data
    df = pd.read_csv(input_path)
    print("Initial shape:", df.shape)
    print(df.head())
    print(df.info())

    # Standardise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Apply data-quality filters and labels
    if "derh_hhage" in df.columns:
        df["derh_hhage"] = df["derh_hhage"].astype(int)
        df = df[(df["derh_hhage"] >= 0) & (df["derh_hhage"] <= 100)]

    if "derh_hhsex" in df.columns:
        df["derh_hhsex"] = df["derh_hhsex"].replace({1: "Male", 2: "Female"})

    if "derh_hsize" in df.columns:
        df = df[df["derh_hsize"] <= 20]

    print("Cleaned shape:", df.shape)
    print(df.describe(include="all").to_string())
    df.describe(include="all").to_csv(output_dir / "summary_statistics.csv")
    df.to_csv(output_dir / "census2022_cleaned.csv", index=False)

    # Frequency tables
    if "derh_hhsex" in df.columns:
        print(df["derh_hhsex"].value_counts())
        df["derh_hhsex"].value_counts().to_csv(output_dir / "household_head_sex_counts.csv")

    if "h05_waterpiped" in df.columns:
        print(df["h05_waterpiped"].value_counts())
        df["h05_waterpiped"].value_counts().to_csv(output_dir / "piped_water_counts.csv")

    # Charts: individual distributions
    plots = [
        ("derh_hhage", "Age Distribution of Household Heads", "Age of Household Head",
         "Count", "age_distribution.png", "hist"),
        ("derh_hsize", "Household Size Distribution", "Household Size",
         "Count", "household_size_distribution.png", "hist_discrete"),
        ("a4_adult_hunger", "Adult Hunger Frequency", "Adult Hunger Code",
         "Count", "adult_hunger_frequency.png", "count"),
        ("a5_child_hunger", "Child Hunger Frequency", "Child Hunger Code",
         "Count", "child_hunger_frequency.png", "count"),
        ("derh_hhsex", "Household Head Sex Distribution", "Household Head Sex",
         "Count", "household_head_sex.png", "count"),
        ("h05_waterpiped", "Access to Piped Water", "Water Piped Code",
         "Count", "piped_water_access.png", "count"),
    ]

    for column, title, xlabel, ylabel, filename, plot_type in plots:
        if column not in df.columns:
            print(f"Skipping chart; column not found: {column}")
            continue
        plt.figure(figsize=(8, 5))
        if plot_type == "hist":
            sns.histplot(df[column], bins=20, kde=True)
        elif plot_type == "hist_discrete":
            sns.histplot(df[column], bins=10, discrete=True)
        else:
            sns.countplot(x=column, data=df)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        save_plot(output_dir, filename)

    # Cross-tabulations and comparison charts
    comparisons = [
        ("derh_hhsex", "h05_waterpiped", "Household Head Sex vs Access to Piped Water",
         "Household Head Sex", "Water Piped Code", "sex_vs_piped_water.png",
         "Access to Piped Water by Household Head Sex"),
        ("derh_hsize", "a4_adult_hunger", "Household Size vs Adult Hunger",
         "Household Size", "Adult Hunger Code", "household_size_vs_adult_hunger.png",
         "Adult Hunger by Household Size"),
        ("derh_hhsex", "a4_adult_hunger", "Household Head Sex vs Adult Hunger",
         "Household Head Sex", "Adult Hunger Code", "sex_vs_adult_hunger.png",
         "Adult Hunger by Household Head Sex"),
        ("h02_maindwelling", "h03_tenure", "Dwelling Type vs Tenure",
         "Dwelling Type Code", "Tenure Code", "dwelling_type_vs_tenure.png",
         "Dwelling Type vs Tenure"),
    ]

    for row_col, hue_col, table_title, xlabel, legend_title, filename, chart_title in comparisons:
        if row_col not in df.columns or hue_col not in df.columns:
            print(f"Skipping comparison; missing columns: {row_col}, {hue_col}")
            continue

        cross_tab = pd.crosstab(df[row_col], df[hue_col])
        print(f"\n{table_title}")
        print(cross_tab)
        cross_tab.to_csv(output_dir / f"{filename.removesuffix('.png')}_crosstab.csv")

        plt.figure(figsize=(10, 6))
        sns.countplot(x=row_col, hue=hue_col, data=df)
        plt.title(chart_title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.legend(title=legend_title)
        save_plot(output_dir, filename)

    # Grouped summary statistics
    if {"derh_hhsex", "derh_hsize"}.issubset(df.columns):
        print("\nMean household size by sex of household head:")
        print(df.groupby("derh_hhsex")["derh_hsize"].mean())
        df.groupby("derh_hhsex")["derh_hsize"].mean().to_csv(
            output_dir / "mean_household_size_by_sex.csv"
        )

    if {"h03_tenure", "derh_hhage"}.issubset(df.columns):
        print("\nAverage age of household head by tenure type:")
        print(df.groupby("h03_tenure")["derh_hhage"].mean())
        df.groupby("h03_tenure")["derh_hhage"].mean().to_csv(
            output_dir / "average_age_by_tenure.csv"
        )

    print(f"\nAnalysis complete. Results saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
