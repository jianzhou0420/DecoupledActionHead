import pandas as pd
import matplotlib.pyplot as plt


data_group1 = {
    'steps': [25, 50, 100, 1000],
    'fid': [29.5561, 26.1112, 24.9252, 21.3906],
    'chr': [12.95, 13.85, 14.55, 10.75],
    'ncrf': [18.06, 12.51, 10.63, 2.39],
    # 'tfr': [31.01, 26.36, 25.19, 13.14]
}
df1 = pd.DataFrame(data_group1)


data_group2 = {
    'steps': [25, 50, 100, 1000],
    'fid': [24.1521, 23.1295, 24.3040, 21.3906],
    'chr': [14.48, 15.99, 15.43, 10.75],
    'ncrf': [9.33, 7.22, 8.94, 2.39],
    # 'tfr': [23.82, 23.21, 24.38, 13.14]
}


data_group1 = {
    'steps': [25, 50, 100],
    'fid': [29.5561, 26.1112, 24.9252],
    'chr': [12.95, 13.85, 14.55],
    'ncrf': [18.06, 12.51, 10.63],
    # 'tfr': [31.01, 26.36, 25.19, 13.14]
}
df1 = pd.DataFrame(data_group1)


data_group2 = {
    'steps': [25, 50, 100],
    'fid': [24.1521, 23.1295, 24.3040],
    'chr': [14.48, 15.99, 15.43],
    'ncrf': [9.33, 7.22, 8.94],
    # 'tfr': [23.82, 23.21, 24.38, 13.14]
}
df2 = pd.DataFrame(data_group2)


# first plot
plt.figure(figsize=(10, 6))
plt.plot(range(len(df1['steps'])), df1['fid'], marker='o', label='fid')
plt.plot(range(len(df1['steps'])), df1['chr'], marker='s', label='chr')
plt.plot(range(len(df1['steps'])), df1['ncrf'], marker='^', label='ncrf')
# plt.plot(range(len(df1['steps'])), df1['tfr'], marker='d', label='tfr')

plt.title('Performance Metrics - First Group of Data')
plt.xlabel('Steps')
plt.ylabel('Values')
plt.legend()

plt.xticks(range(len(df1['steps'])), df1['steps'])
plt.grid(True)
plt.tight_layout()
plt.savefig('A_first_plot.png')
plt.close()

# second plot
plt.figure(figsize=(10, 6))

plt.plot(range(len(df2['steps'])), df2['fid'], marker='o', label='fid')
plt.plot(range(len(df2['steps'])), df2['chr'], marker='s', label='chr')
plt.plot(range(len(df2['steps'])), df2['ncrf'], marker='^', label='ncrf')
# plt.plot(range(len(df2['steps'])), df2['tfr'], marker='d', label='tfr')

plt.title('Performance Metrics - Second Group of Data')
plt.xlabel('Steps')
plt.ylabel('Values')
plt.legend()

plt.xticks(range(len(df2['steps'])), df2['steps'])
plt.grid(True)
plt.tight_layout()
plt.savefig('A_second_plot.png')
plt.close()
