# Constructing a lookup for meaning of columns
import pandas as pd

df_desc = pd.read_excel("LCDataDictionary.xlsx", nrows = 151)
df_desc["LoanStatNew"] = df_desc["LoanStatNew"].apply(lambda x: x.rstrip()) # some cells have trailing whitespace
df_desc["LoanStatNew"] = df_desc["LoanStatNew"].replace({'verified_status_joint':'verification_status_joint'}) # match name in df
desc = {stat:description for stat,description in zip(df_desc["LoanStatNew"], df_desc["Description"])}

if __name__ == "__main__":
	# example
	print(desc["acc_now_delinq"])

	# checking to make sure columns and descriptions match 1-to-1
	df = pd.read_csv("train.csv")
	print(set(desc.keys()).difference(set(df.columns.values))) # keys in desc which do not have a match in df
	print(set(df.columns.values).difference(set(desc.keys()))) # keys (columns) in df which do not have a match in desc 