import os
import numpy as np
def read_ate_and_error_values(file_path):
    ate_and_error_values = []
    with open(file_path, 'r') as f:
        for line in f:
            ate, error = map(float, line.strip().split())
            ate_and_error_values.append((ate, error))
    return ate_and_error_values

def read_values_from_txt(file_path):
    values = []
    with open(file_path, 'r') as f:
        for line in f:
            values.append(float(line.strip()))
    return values


results_folder_name = 'ate_and_error_results'

# Read ATE and error values for both models
ate_and_error_file_model1 = os.path.join(results_folder_name, 'ate_and_error_values5.txt')
ate_and_error_values_model1 = read_ate_and_error_values(ate_and_error_file_model1)

ate_and_error_file_model2 = os.path.join(results_folder_name, 'ate_and_error_values6.txt')
ate_and_error_values_model2 = read_ate_and_error_values(ate_and_error_file_model2)

# Read MSE values for both models
mse_file_model1 = os.path.join(results_folder_name, 'mse_values1.txt')
mse_values_model1 = read_values_from_txt(mse_file_model1)

mse_file_model2 = os.path.join(results_folder_name, 'mse_values2.txt')
mse_values_model2 = read_values_from_txt(mse_file_model2)

# Read actual ATE values
actual_ate_values = read_values_from_txt('ate_values.txt')

# Calculate bounds for both models
bounds_model1 = [x[1] for x in ate_and_error_values_model1]
bounds_model2 = [x[1] for x in ate_and_error_values_model2]

# Calculate the percentage by which bounds are tighter for Model 2
percentage_tighter_bounds = ((np.mean(bounds_model2) - np.mean(bounds_model1)) / np.mean(bounds_model2)) * 100

print(f"Model 1 has bounds that are {percentage_tighter_bounds:.2f}% tighter than Model 2.")

# Calculate the percentage by which MSE is lower for Model 1
percentage_lower_mse = ((np.mean(mse_values_model2) - np.mean(mse_values_model1)) / np.mean(mse_values_model2)) * 100

#Print Sum of MSE Values
print(f"Sum of MSE Values for Model 1: {np.sum(mse_values_model1)}")
print(f"Sum of MSE Values for Model 2: {np.sum(mse_values_model2)}")

print(f"Model 1 has MSE that is {percentage_lower_mse:.2f}% lower than Model 2.")

def compute_metrics(ate_and_error_values, actual_ate_values):
    mae = np.mean([abs(x[0] - actual_ate) for x, actual_ate in zip(ate_and_error_values, actual_ate_values)])
    mse = np.mean([(x[0] - actual_ate) ** 2 for x, actual_ate in zip(ate_and_error_values, actual_ate_values)])
    mpe = np.mean(
        [abs((x[0] - actual_ate) / actual_ate) for x, actual_ate in zip(ate_and_error_values, actual_ate_values)]) * 100

    return mae, mse, mpe


# Calculate metrics for both models
mae_model1, mse_model1, mpe_model1 = compute_metrics(ate_and_error_values_model1, actual_ate_values)
mae_model2, mse_model2, mpe_model2 = compute_metrics(ate_and_error_values_model2, actual_ate_values)

print(f"Model 1: Mean Absolute Error in ATE estimation: {mae_model1}")
print(f"Model 1: Mean Squared Error in ATE estimation: {mse_model1}")
print(f"Model 1: Mean Percentage Error in ATE estimation: {mpe_model1}%")

print(f"Model 2: Mean Absolute Error in ATE estimation: {mae_model2}")
print(f"Model 2: Mean Squared Error in ATE estimation: {mse_model2}")
print(f"Model 2: Mean Percentage Error in ATE estimation: {mpe_model2}%")

