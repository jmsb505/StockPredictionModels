import pandas as pd

# Sample DataFrames
df1 = pd.DataFrame({
    'date': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04'],
    'value1': [1, 2, 3, 4]
})

df2 = pd.DataFrame({
    'date': ['2021-01-01', '2021-01-03', '2021-01-04'],
    'value2': [10, 20, 30]
})

df1['date'] = pd.to_datetime(df1['date'])
df2['date'] = pd.to_datetime(df2['date'])

merged = df1.merge(df2, on='date', how='left')
merged['ffill'] = merged['value2'].ffill()
merged['cumsum'] = merged.groupby('ffill')['value1'].cumsum()

print(merged)

final = merged.loc[merged['value2'].notna(), ['date', 'cumsum', 'value2']].reset_index(drop=True)
final.columns = ['date', 'value1_aggregated', 'value2']

print(final)
