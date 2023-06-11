import pandas as pd
import numpy as np

df = pd.read_excel('synthetic_data_with_infotreat4.xlsx')
Y0_df = df['Y0'].values.reshape(-1, 20)
Y1_df = df['Y1'].values.reshape(-1, 20)
T_df = df['T'].values.reshape(-1, 20)

individual_treatment_effect = Y1_df - Y0_df

ATE = np.mean(individual_treatment_effect)

treated_individuals = individual_treatment_effect[T_df == 1]
ATT = np.mean(treated_individuals)

print("Average Treatment Effect (ATE):", ATE)
print("Average Treatment Effect on the Treated (ATT):", ATT)
