# -*- coding: utf-8 -*-
"""
"""
# load package and data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import time
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import seaborn as sns

# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPUs Available: ", physical_devices)
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("No GPU found. Using CPU instead.")

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

gamelabel=pd.read_csv('gamelabel_1016_new.csv')

Pixel_features=pd.read_csv('Poly_47to98_concatenated_new.csv')
   

gamelabel=gamelabel.iloc[:150*4320]    # default:gamelabel.iloc[:152*4320] 
Pixel_features=Pixel_features.iloc[:150*4320]   # default:gamelabel.iloc[:152*4320]

#data processing
train_split = 109*24*180   #use May to August as the training data, !!!!!!# defalut: (31+30+31+31)*24*180 
learning_rate = 0.001
batch_size = 288*12  #4320
epochs = 50


#-------------Add calendar features------------------
new_datetime = gamelabel[['datatime']].copy()

new_datetime.loc[:, 'datetime'] = pd.to_datetime(new_datetime['datatime'])


new_datetime['weekday'] = new_datetime['datetime'].dt.dayofweek  # Monday=0, Sunday=6
new_datetime['hour'] = new_datetime['datetime'].dt.hour

new_datetime['normalized_dayofweek'] = new_datetime['weekday'] / 6
# Normalize hour
new_datetime['normalized_hour'] = new_datetime['hour'] / 23


Pixel_features['nor_dayofweek'] = new_datetime['normalized_dayofweek'] 
Pixel_features['nor_hour'] = new_datetime['normalized_hour']

#---------------Maksing---------------
# new_Pixel_features = Pixel_features.copy()
# Pixel_features_mask = new_Pixel_features.applymap(lambda x: 0 if x == -1 else 1)

#Pixel_features.replace(-1, 0, inplace=True) # change -1 in dataframe to 0

#---------------Normalization---------------
den_cap=30
den_max=162
vol_cap=1800
    
# for col in range(0,28,2):  #density
#     Pixel_features.iloc[:, col] = Pixel_features.iloc[:, col].apply(lambda x: (x - den_mean) / den_std if x != -1 else x)
    
# for col in range(1,28,2): # volume
#     Pixel_features.iloc[:, col] = Pixel_features.iloc[:, col].apply(lambda x: (x - vol_mean) / vol_std if x != -1 else x)
    
for col in range(0,28,2):  #density
    Pixel_features.iloc[:, col] = Pixel_features.iloc[:, col].apply(lambda x: x / den_cap if x != -1 else x)
    
for col in range(1,28,2): # volume
    Pixel_features.iloc[:, col] = Pixel_features.iloc[:, col].apply(lambda x: x / 900 if x != -1 else x)  
    
# adjacency matrix    
Pixel_features.iloc[:, 28:2*14+49*2] = Pixel_features.iloc[:, 28:2*14+49*2].replace({0: -2, 1: 2})   

for col in range(2*14+49*2,322,2):  #time dimention
    Pixel_features.iloc[:, col] = Pixel_features.iloc[:, col].apply(lambda x: x*180 if x != -1 else x)

for col in range(2*14+49*2+1,224,2):  #distance dimention
    Pixel_features.iloc[:, col] = Pixel_features.iloc[:, col].apply(lambda x: x*2.5 if x != -1 else x)
    
for col in range(224+1,322,2):  #distance dimention
    Pixel_features.iloc[:, col] = Pixel_features.iloc[:, col].apply(lambda x: x/0.49 if x != -1 else x)
    
Pixel_features.replace(-1, 0, inplace=True) # change -1 in dataframe to 0

Pixel_features.iloc[:, 28:2*14+49*2] = Pixel_features.iloc[:, 28:2*14+49*2].replace({-2: -1, 2: 1})   
Normalized_features=Pixel_features
#-----------------------------------
#-----------------------------------


# Training dataset
train_data = Normalized_features.iloc[0 : train_split]
train_label = gamelabel.iloc[0 : train_split]
test_data = Normalized_features.iloc[train_split:]
test_label = gamelabel.iloc[train_split:]
test_data.reset_index(drop=True, inplace=True)
test_label.reset_index(drop=True, inplace=True)

#train_mask_data = Pixel_features_mask.iloc[0 : train_split] 
#test_mask_data = Pixel_features_mask.iloc[train_split:] 

# Training dataset
TIME_STEPS = 360
step=15
# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS, step_interval=step):
    output = []
    for j in range(int(len(values)/4320)):
        for i in range((4320*j+10*180), (4320*(j+1)-time_steps), step_interval):
            output.append(values[i : (i + time_steps)])
    return np.stack(output)


XX_train = create_sequences(train_data.values)
YY_train = np.array([train_label['label'][ind] for day in range(int(len(train_data)/4320)) for ind in range((4320*day+10*180+TIME_STEPS), (4320*(day+1)), step)])
YY_train_onehot=to_categorical(YY_train, 4)

#XX_mask_train = create_sequences(train_mask_data.values)   # mask
print("Training input shape and label shape: ", XX_train.shape, YY_train.shape)


# Test dataset
XX_test = create_sequences(test_data.values)
YY_test = np.array([test_label['label'][ind] for day in range(int(len(test_data)/4320)) for ind in range((4320*day+10*180+TIME_STEPS), (4320*(day+1)), step)])
YY_test_onehot=to_categorical(YY_test, 4)

#XX_mask_test = create_sequences(test_mask_data.values)   # mask
print("Training input shape and label shape: ", XX_test.shape, YY_test.shape)

def class_accuracy(class_id):
    def acc(y_true, y_pred):
        # Find the true labels for the given class
        y_true_class = tf.equal(tf.argmax(y_true, axis=-1), class_id)
        # Find the predicted labels for the given class
        y_pred_class = tf.equal(tf.argmax(y_pred, axis=-1), class_id)

        # Convert to float to calculate accuracy
        correct_predictions = tf.reduce_sum(tf.cast(y_true_class & y_pred_class, tf.float32))
        total_class_count = tf.reduce_sum(tf.cast(y_true_class, tf.float32))
        
        # Avoid division by zero by adding epsilon
        acc = correct_predictions / (total_class_count + tf.keras.backend.epsilon())
        return acc
    acc.__name__ = f'accuracy_class_{class_id}'
    return acc

# Set up MirroredStrategy for multi-GPU usage
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2"])

with strategy.scope():
    # Model definition
    inputs = keras.layers.Input(shape=(XX_train.shape[1], XX_train.shape[2]))
    simple_rnn_out2 = keras.layers.LSTM(96, dropout=0.5, recurrent_dropout=0.5)(inputs)
    outputs = keras.layers.Dense(4, activation='softmax')(simple_rnn_out2)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Metrics for each class
    num_classes = 4
    precision_metrics = [Precision(class_id=i, name=f'precision_class_{i}') for i in range(num_classes)]
    recall_metrics = [Recall(class_id=i, name=f'recall_class_{i}') for i in range(num_classes)]
    acc_metrics = [class_accuracy(i) for i in range(num_classes)]

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[*precision_metrics, *recall_metrics,*acc_metrics]
    )

    model.summary()

    path_checkpoint = "model_checkpoint.weights.h5"

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    # Training
    start_time = time.time()

    weights = {0: 1.7, 1: 2.5, 2:1.8, 3:1}
    history = model.fit(
        XX_train,
        YY_train_onehot,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(XX_test, YY_test_onehot),
        callbacks=[modelckpt_callback],
        shuffle=False,
        class_weight=weights,
    )

    lstm_layer = model.layers[1]  # Accessing the LSTM layer (it's the second layer)
    lstm_weights = lstm_layer.get_weights()

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

RNN_kernel=lstm_weights[0]
RNN_rekernel=lstm_weights[1]
dfRNN_kernel = pd.DataFrame(RNN_kernel)
dfRNN_rekernel = pd.DataFrame(RNN_rekernel)
dfRNN_kernel.to_csv('poly_fRNN_kernel_t3.csv', index=False) 
dfRNN_rekernel.to_csv('poly_fRNN_rekernel_t3.csv', index=False) 


########################## ALERT EVENT ON 08/14################
# Collect predictions
preds = model.predict(XX_test[864:1728])
preds_classes=preds.argmax(axis=1)
true_label=YY_test[864:1728]
print(classification_report(true_label, preds_classes))
# Flatten the list of predictions
#predictions_1 = np.concatenate(predictions, axis=0)
df_pred = pd.DataFrame(preds)
df_pred.to_csv('pred_poly_multigpus_4CLASSES_10AM_t3.csv', index=False) 


########################## AVERAGE ALERT EVENT ################
#changes = np.diff(YY_test) == 1
#indices = np.where(changes)[0] + 1  # Add 1 because diff reduces the length by 1
indices = []
for i in range(1, len(YY_test)):
    if YY_test[i] == 1 and YY_test[i-1] != 1:
        indices.append(i)

Count_event=[]
for indx in range(len(indices)):
    predictions_temp = model.predict(XX_test[indices[indx]-24:indices[indx]+60])
    preds_temp_classes=predictions_temp.argmax(axis=1)
    new_label = np.where(preds_temp_classes==1, 1, 0)
    Count_event.append(np.squeeze(new_label))
    

Count_event_array = np.stack(Count_event, axis=0)
Count_final = np.sum(Count_event_array, axis=0)  
divided_array = Count_final / len(indices)

df_divided_array = pd.DataFrame(divided_array)
df_divided_array.to_csv('PEPE_poly_multigpus_4CLASSES_10AM_t3.csv', index=False) 

