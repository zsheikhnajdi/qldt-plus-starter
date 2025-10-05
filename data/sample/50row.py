import pandas as pd
df = pd.read_csv("data/raw/pima-indians-diabetes.data.csv", header=None)
df.head(50).to_csv("data/sample/pima-indians-diabetes.data.csv", index=False, header=False)
