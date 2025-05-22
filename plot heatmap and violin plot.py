# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from matplotlib.colors import TwoSlopeNorm
#------------------------------HEATMAP-----------------------------------------------------
hidden_units=96


# Z-encoding + vanilla LSTM
RNN_f_ker=pd.read_csv('poly_fRNN_kernel_t3.csv')
df_f_kernel = RNN_f_ker.iloc[:,:hidden_units]

#X-encoding 9by12 + vanilla LSTM
RNN_9by12_ker=pd.read_csv('9by12_RNN_kernel_t3.csv')
df_9by12_kernel = RNN_9by12_ker.iloc[:,:hidden_units]


# Load Sparse LSTM the weights
data = np.load("sparse_lstm_weights.npz", allow_pickle=True)

k_indices = tf.constant(data['k_indices'], dtype=tf.int64)
k_values = tf.Variable(data['k_values'], dtype=tf.float32)
k_values_np = k_values.numpy()
k_shape = tuple(data['k_shape'])



vmin = min(k_values_np.min(), df_9by12_kernel.min().min(), df_f_kernel.min().min())
vmax = max(k_values_np.max(), df_9by12_kernel.max().max(), df_f_kernel.max().max())

# Create a custom diverging palette with white at zero
cmap = sns.diverging_palette(240, 10, sep=20, as_cmap=True)

# Create a normalizer to ensure 0 is mapped to white
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)


plt.figure(figsize=(12, 8))
sns.heatmap(df_9by12_kernel, cmap=cmap, norm=norm, cbar=True)    ## icefire, vlag
plt.title('Pixel-based (9by12) Vanilla LSTM Kernel Weights Input Gate Heatmap')
plt.xlabel('LSTM Input Gate Units',fontsize=24)
plt.ylabel('Input Features',fontsize=24)
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df_f_kernel, cmap=cmap, norm=norm, cbar=True)   ## icefire, vlag
plt.title('Puzzle-based Vanilla LSTM Kernel Weights Input Gate Heatmap')
plt.xlabel('LSTM Input Gate Units',fontsize=24)
plt.ylabel('Input Features',fontsize=24)
plt.show()



#Z encoding sparse LSTM

def plot_input_gate_weights_from_data(k_indices, k_values, k_shape, units):
    """
    Plots the input gate sparse weights heatmap based on raw sparse matrix components.
    
    Parameters:
    - k_indices: tf.Tensor or np.ndarray of shape (N, 2)
    - k_values: tf.Tensor or np.ndarray of shape (N,)
    - k_shape: tuple of (input_dim, 4 * units)
    - units: number of LSTM units (used to slice the input gate block)
    """
    if isinstance(k_indices, tf.Tensor):
        k_indices = k_indices.numpy()
    if isinstance(k_values, tf.Variable) or isinstance(k_values, tf.Tensor):
        k_values = k_values.numpy()
    
    input_dim, _ = k_shape
    input_gate_weights = np.zeros((input_dim, units), dtype=np.float32)

    for (i, j), v in zip(k_indices, k_values):
        if j < units:  # Input gate block
            input_gate_weights[i, j] = v

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(input_gate_weights, cmap=cmap, norm=norm, cbar=True)
    plt.title('Sparse LSTM Input Gate Weight Heatmap')
    plt.xlabel('LSTM Input Gate Units')
    plt.ylabel('Input Features')
    plt.tight_layout()
    plt.show()


plot_input_gate_weights_from_data(k_indices, k_values, k_shape, units=96)






#------------------------------Violin Plot------------------------------------------------
# Z-encoding + vanilla LSTM

RNN1_ker1=pd.read_csv('poly_fRNN_kernel_t3.csv')
RNN1_re_ker1=pd.read_csv('poly_fRNN_rekernel_t3.csv')



G1B11 =  RNN1_ker1.iloc[:, :int(RNN1_ker1.shape[1]/4)].values.ravel()


print("Mean", np.mean(G1B11))
print("Std", np.std(G1B11))
# Create a DataFrame for easier plotting
data_G1K1 = pd.DataFrame({
    '$U_i$': pd.Series(G1B11)  # Series allows for different lengths
})
# Melt the DataFrame to a long format
data_melted_G1K = data_G1K1.melt(var_name='Group', value_name='Values')
# Create the violin plot
plt.figure(figsize=(8, 6))
ax=sns.violinplot(x='Group', y='Values', data=data_melted_G1K)
# Display the plot
plt.title("Violin Plot of Input Gate kernel weights V-Z Case")  #puzzle basic LSTM
plt.ylim(-0.4, 0.4)
plt.xlabel('Group', fontsize=20)
plt.ylabel('Values', fontsize=20)
ax.tick_params(axis='y', labelsize=15)  # Adjust y-axis tick labels
ax.tick_params(axis='x', labelsize=20)  # Adjust x-axis tick labels
plt.show()


#X-encoding 9by12 + vanilla LSTM
RNN1_ker_912=pd.read_csv('9by12_RNN_kernel_t3.csv')

G1B1_912 =  RNN1_ker_912.iloc[:, :int(RNN1_ker_912.shape[1]/4)].values.ravel()


print("Mean", np.mean(G1B1_912))
print("Std", np.std(G1B1_912))
# Create a DataFrame for easier plotting
data_G1K_912 = pd.DataFrame({
    '$U_i$': pd.Series(G1B1_912)  # Series allows for different lengths
})
# Melt the DataFrame to a long format
data_melted_G1K_912 = data_G1K_912.melt(var_name='Group', value_name='Values')
# Create the violin plot
plt.figure(figsize=(8, 6))
ax2=sns.violinplot(x='Group', y='Values', data=data_melted_G1K_912)
# Display the plot
plt.title("Violin Plot of Input Gate kernel weights V-X (9by12) Case")  #puzzle basic LSTM
plt.ylim(-0.4, 0.4)
plt.xlabel('Group', fontsize=20)
plt.ylabel('Values', fontsize=20)
ax2.tick_params(axis='y', labelsize=15)  # Adjust y-axis tick labels
ax2.tick_params(axis='x', labelsize=20)  # Adjust x-axis tick labels
plt.show()



#Z-encoding 9by12 + Sparse LSTM
# Load the weights
data = np.load("sparse_lstm_weights.npz", allow_pickle=True)
k_indices = tf.constant(data['k_indices'], dtype=tf.int64)
k_values = tf.Variable(data['k_values'], dtype=tf.float32)
k_shape = tuple(data['k_shape'])

# Parameters
units = 96  # Set the number of LSTM units manually

# Define the input feature block ranges (same as in your kernel mask)
input_blocks = [
    (0, 28),                # Block 1
    (28, 28 + 98),          # Block 2
    (28 + 98, 28 + 98 + 196)  # Block 3
]

# Convert to NumPy arrays if needed
k_indices_np = k_indices.numpy()
k_values_np = k_values.numpy()

# Group weights by input block for input gate (first 96 columns)
violin_data = []

for (i, j), v in zip(k_indices_np, k_values_np):
    if j < units:
        for b, (start_row, end_row) in enumerate(input_blocks):
            if start_row <= i < end_row:
                violin_data.append({"Value": v, "Group": f"$B^u_{{{b+1}}}$"})
                break

df_violin = pd.DataFrame(violin_data)

# Plot violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(data=df_violin, x="Group", y="Value", palette="Set2")
plt.title("Violin Plot of Input Gate Kernel Weights for Blocks $B^u_1$, $B^u_2$, $B^u_3$")
plt.tight_layout()
plt.show()