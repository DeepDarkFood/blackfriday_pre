import pandas as pd
pd.set_option('display.max_columns', None)


chunks = pd.read_csv('./data/BlackFriday.csv',iterator=True)
for chunk in chunks:
    pass
