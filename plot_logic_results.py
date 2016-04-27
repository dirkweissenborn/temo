import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


df = pd.read_csv(open('./summary.txt'), sep='\t', header=None)
df.columns = ['Cell', 'k', 'Train', 'Dev', 'Test']
df["k"] = df["k"].astype(int)
print(df)
sns.axlabel("Memory Size", "Accuracy")


sns.set_context("poster")
plt.figure(figsize=(5, 4.2))

sns.set_style("white")
sns.set_style("ticks")


sns_plot = sns.pointplot(x="k", y="Test", data=df, hue="Cell",
                         markers=["o", "x"], linestyles=["-", "--"])

plt.legend(loc='lower right')
plt.ylim(0.5, 1.0)


plt.savefig("summary.pdf")
