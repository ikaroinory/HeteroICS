import pandas as pd
from matplotlib import pyplot as plt

column_name = [
    'FIT101', 'LIT101', ' MV101', 'P101', 'P102',
    ' AIT201', 'AIT202', 'AIT203', 'FIT201', ' MV201', ' P201', ' P202', 'P203', ' P204', 'P205', 'P206',
    'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', ' MV303', 'MV304', 'P301', 'P302',
    'AIT401', 'AIT402', 'FIT401', 'LIT401', 'P401', 'P402', 'P403', 'P404', 'UV401',
    'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504',
    'P501', 'P502', 'PIT501', 'PIT502', 'PIT503',
    'FIT601', 'P601', 'P602', 'P603',
    'Attack'
]

df_actual = pd.read_csv("actual.csv", names=column_name)
df_predict = pd.read_csv("predict.csv", names=column_name)

fig, axs = plt.subplots(3, 1, figsize=(10, 8))

axs[0].plot(df_actual['FIT401'].iloc[20000:30000])
axs[0].set_title('Actual (FIT401)')

axs[1].plot(df_predict['FIT401'].iloc[20000:30000])
axs[1].set_title('Predict (FIT401)')

axs[2].plot(df_predict['Attack'].iloc[20000:30000])
axs[2].set_title('Attack')

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(10, 8))
