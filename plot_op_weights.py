import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


#df = pd.read_csv(open('./op_weights.txt.tmp'), sep='\t', header=0)
df = pd.read_csv(open('./op_weights.txt'), sep='\t', header=0)

#print(df)
df_sum = df.sum(axis=1)

for col in ["max", "keep", "replace", "mul", "min", "diff", "forget"]:
    df[col] = df[col] / df_sum

#print(df)
ax = sns.heatmap(df, linewidths=.5, cbar=False, square=True)
plt.xticks(rotation=-30)
plt.yticks(rotation=0)

plt.savefig("op_weights.pdf")
plt.savefig("op_weights.png")

