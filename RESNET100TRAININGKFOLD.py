import time
import tqdm
for i in tqdm.tqdm(range(4500)):
    time.sleep(1)


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import datetime
import seaborn as sns

import DATALOADER
import init_models

model_id = 'KFOLDRN101'+datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
path_dir = 'train_results'
#path_dir = 'test'
base_path = os.path.join(path_dir, model_id)
path_plots = os.path.join(base_path, model_id+'plot_')
path_checkpoint = os.path.join(base_path, model_id+'_checkpoint')
path_model = os.path.join(base_path, model_id+'_model.h5')  
path_log = os.path.join(base_path, model_id+'_log.csv')  
path_y_true = os.path.join(base_path, model_id+'_y_true')  
path_y_pred = os.path.join(base_path, model_id+'_y_pred')  
path_cm = os.path.join(base_path, model_id+'_cm')  
path_results = os.path.join(base_path, model_id+'_results.txt')


#General training
batch_size = 32
n_epochs = 20
test_size = 0.5
random_state = 0
shuffle = True

#Dynamic Learning rate
factor = 0.1
patience = 5
min_lr=1e-6

#Optimizer
learning_rate=1e-2
momentum=0.9
decay = 0.01

#Kfold
n_splits = 5

if not os.path.exists(base_path):
    os.makedirs(base_path)
    
if not os.path.exists(path_checkpoint):
    os.makedirs(path_checkpoint)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

path = 'Data_paths.csv'
df = pd.read_csv(path)
df = df[df.moa != 'DMSO']

#df = df.groupby('moa').sample(n=10, random_state=0).reset_index(drop=True)

moa_one_hot_encoder = sklearn.preprocessing.OneHotEncoder(sparse=False)
moa_one_hot_encoder.fit(df['moa'].to_numpy().reshape(-1, 1))


skf = sklearn.model_selection.StratifiedKFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)


input_tensor = tf.keras.Input(shape=(68,68,3))

rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor = factor, patience = patience, min_lr=min_lr)
check_point = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint ,verbose=1, monitor = "val_accuracy", mode = "max", save_best_only=True)
csv_logger = tf.keras.callbacks.CSVLogger(path_log, append = True, separator=',')

histories = []

for idx_fold, (idx_train, idx_test) in enumerate(skf.split(df['path'], df['moa'])):
    print(f"\n\nKfold iteration: {idx_fold+1}/{n_splits}\n")
    df_train = df.loc[idx_train]
    df_test = df.loc[idx_test]

    model_res = tf.keras.applications.ResNet101V2(include_top = False, weights = None, input_tensor = input_tensor)


    # for layer in model_res.layers[:143]:
    #     layer.trainable = True
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.RandomFlip())
    model.add(tf.keras.layers.RandomRotation(0.1))
    model.add(model_res)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(12, activation = 'softmax'))

    train_dataloader = DATALOADER.DataLoader(df = df_train, model = model, one_hot_encoder = moa_one_hot_encoder, batch_size=batch_size, input_size=(68,68,3), shuffle = True)
    test_dataloader = DATALOADER.DataLoader(df = df_test, one_hot_encoder = moa_one_hot_encoder, batch_size = batch_size, input_size=(68,68,3), shuffle = True)

    
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate, momentum=momentum, decay = decay),metrics=['accuracy'])

    history=model.fit(train_dataloader, validation_data=test_dataloader, epochs= n_epochs, callbacks = [check_point, rlrop, csv_logger])
    histories.append(history)

    indices = np.arange(len(df))
    X_train, X_test, y_train, y_test, idx_train, idx_test = sklearn.model_selection.train_test_split(df['path'], df['moa'], indices, test_size = test_size, random_state=random_state, shuffle = shuffle, stratify=df['moa'])
    df_valid = df.loc[idx_test]
    validation_dataloader = DATALOADER.DataLoader(df = df_valid, one_hot_encoder = moa_one_hot_encoder, batch_size = batch_size, input_size=(68,68,3))

    y_pred = model.predict(validation_dataloader)
    #Confusion Matrix
    y_true = df_valid.moa[0:len(y_pred)]
    y_pred = moa_one_hot_encoder.inverse_transform(y_pred)
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')

    np.save(f"{path_y_true}_fold_{idx_fold+1}.npy", y_true)
    np.save(f"{path_y_pred}_fold_{idx_fold+1}.npy", y_pred)
    np.save(f"{path_cm}_fold_{idx_fold+1}.npy", cm)
    plt.savefig(f"{path_cm}_fold_{idx_fold+1}.png", bbox_inches = 'tight')




fig = plt.figure(figsize=(10,10))
for hist in histories:
    plt.plot(hist.history['accuracy'])
plt.title('model train accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['Kfold iteration '+str(i) for i in range(n_epochs)], loc='upper left')
plt.savefig(path_plots+'acc.png', bbox_inches = 'tight')

fig = plt.figure(figsize=(10,10))
for hist in histories:
    plt.plot(hist.history['val_accuracy'])
plt.title('model validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['Kfold iteration '+str(i) for i in range(n_epochs)], loc='upper left')
plt.savefig(path_plots+'val_acc.png', bbox_inches = 'tight')

fig = plt.figure(figsize=(10,10))
for hist in histories:
    plt.plot(hist.history['loss'])
plt.title('model validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Kfold iteration '+str(i) for i in range(n_epochs)], loc='upper left')
plt.savefig(path_plots+'loss.png', bbox_inches = 'tight')

fig = plt.figure(figsize=(10,10))
for hist in histories:
    plt.plot(hist.history['val_loss'])
plt.title('model validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Kfold iteration '+str(i) for i in range(n_epochs)], loc='upper left')
plt.savefig(path_plots+'val_loss.png', bbox_inches = 'tight')

n = len(histories)
np_size= (n,n_epochs)
acc = np.empty(np_size)
val_acc = np.empty(np_size)
loss = np.empty(np_size)
val_loss = np.empty(np_size)
for index, hist in enumerate(histories):
    acc[index, :] = hist.history['accuracy']
    val_acc[index, :] = hist.history['val_accuracy']
    loss[index, :] = hist.history['loss']
    val_loss[index, :] = hist.history['val_loss']

mu_acc = np.mean(acc, axis = 0)
mu_val_acc = np.mean(val_acc, axis = 0)
mu_loss = np.mean(loss, axis = 0)
mu_val_loss = np.mean(val_loss, axis = 0)

fig = plt.figure(figsize=(10,10))
plt.plot(mu_acc)
plt.plot(mu_val_acc)
plt.title('average model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(path_plots+'average_acc.png', bbox_inches = 'tight')

fig = plt.figure(figsize=(10,10))
plt.plot(mu_loss)
plt.plot(mu_val_loss)
plt.title('average model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(path_plots+'average_loss.png', bbox_inches = 'tight')




with open(path_results, 'w+') as f:
    f.write(f"\nGENERAL TRAINING\n")
    f.write(f"batch_size:   {batch_size}\n")
    f.write(f"n epochs:     {n_epochs}\n")
    f.write(f"test size:    {test_size}\n")
    f.write(f"random state: {random_state}\n")
    f.write(f"shuffle:      {shuffle}\n")
    f.write(f"n train data: {len(train_dataloader)}\n")
    f.write(f"n test data:  {len(test_dataloader)}\n")
    
    f.write(f"\nDYNAMIC LEARNING RATE\n")
    f.write(f"factor:       {factor}\n")
    f.write(f"patience:     {patience}\n")
    f.write(f"min_lr:       {min_lr}\n")

    f.write(f"\nOPTIMIZER\n")
    f.write(f"learning rate:{learning_rate}\n")
    f.write(f"momentum:     {momentum}\n")
    f.write(f"decay:        {decay}\n")
    
    f.write(f"\nKFOLD\n")
    f.write(f"n splits:     {n_splits}\n")    
    

    
    f.write(f"\nRESULTS\n")
    f.write(f"accuracy:            {sklearn.metrics.accuracy_score(y_true, y_pred)*100:.2f}%\n")



def save_model_summary(summary):
    with open(path_results, 'a') as f:
        f.write('\n\n\n\n\n')
        print(summary, file = f)
        
def save_model_summary(summary):
    with open(path_results, 'a') as f:
        f.write('\n\n\n\n\n')
        print(summary, file = f)
        
        
model.summary(print_fn = save_model_summary)
model.get_layer('resnet50v2').summary(print_fn = save_model_summary)
model.save(path_model)
    
    