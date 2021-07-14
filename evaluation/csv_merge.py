import numpy as np
import pandas as pd

df1 = pd.read_csv('../datasets/100_rows.csv', encoding='utf-8')
df2 = pd.read_csv('scores/entailment_scores.csv', encoding='utf-8')
df2 = df2.dropna()

entailments = []
neutrals = []
minuses = []
contradictions = []

for i in range(100):
  entailments.append(df2['E'][i])
  neutrals.append(df2['N'][i])
  minuses.append(df2['-'][i])
  contradictions.append(df2['C'][i])

# print(df1.head())

df1.insert(8, "E", entailments, True)
df1.insert(9, "N", neutrals, True)
df1.insert(10, "-", minuses, True)
df1.insert(11, "C", contradictions, True)

print(df1.head())

df1.to_csv('scores/final_evaluation_scores.csv')