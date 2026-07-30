import matplotlib.pyplot as plt
import seaborn as sns

# Load Iris Dataset
df = sns.load_dataset("iris")

# Set overall style
sns.set_theme(
    style="whitegrid",
    context="notebook"
)

# Create Pair Plot
pair_plot = sns.pairplot(
    df,
    hue="species",
    palette="Set1",
    diag_kind="kde",
    markers=["o", "s", "D"],
    plot_kws={
        "alpha": 0.7,
        "s": 55,
        "edgecolor": "white",
        "linewidth": 0.5
    }
)

# Add main title
pair_plot.figure.suptitle(
    "🌸 Iris Flower Dataset Analysis",
    fontsize=20,
    fontweight="bold",
    y=1.03
)

# Add subtitle
pair_plot.figure.text(
    0.5,
    1.00,
    "Relationship Between Sepal and Petal Measurements",
    ha="center",
    fontsize=11
)

# Adjust spacing
pair_plot.figure.subplots_adjust(
    top=0.92,
    wspace=0.08,
    hspace=0.08
)

# Show Pair Plot
plt.show()


# -------------------------------
# Correlation Heatmap
# -------------------------------

# Select only numerical columns
numeric_df = df.select_dtypes(include="number")

# Create correlation matrix
correlation = numeric_df.corr()

# Create Heatmap
plt.figure(figsize=(9, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=1,
    square=True,
    cbar_kws={"label": "Correlation Value"}
)

# Heatmap title
plt.title(
    "🔥 Correlation Heatmap of Iris Features",
    fontsize=16,
    fontweight="bold",
    pad=15
)

plt.tight_layout()
plt.show()