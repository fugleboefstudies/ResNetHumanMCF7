import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
import datetime

import DATALOADER
import init_models

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

moa_one_hot_encoder = sklearn.preprocessing.OneHotEncoder(sparse=False)
moa_one_hot_encoder.fit(df['moa'].to_numpy().reshape(-1, 1))

indices = np.arange(len(df))
X_train, X_test, y_train, y_test, idx_train, idx_test = sklearn.model_selection.train_test_split(df['path'], df['moa'], indices, test_size = 0.5, random_state=0, shuffle = True, stratify=df['moa'])

df_train = df.loc[idx_train]
df_test = df.loc[idx_test]


for i in [0,1,2,3]:
    model_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    base_path = 'train_results'
    path_checkpoint = os.path.join(base_path, model_id+'_checkpoint')
    path_model = os.path.join(base_path, model_id+'_model.h5')  
    path_log = os.path.join(base_path, model_id+'_log.csv')  
    
    model = init_models.init_default_model(i)

    #train_dataloader = DataLoader(df = df_train, model = model, one_hot_encoder = moa_one_hot_encoder, learning_rate_schedule, batch_size=32, input_size=(68,68,3))
    train_dataloader = DATALOADER.DataLoader(df = df_train, model = model, one_hot_encoder = moa_one_hot_encoder, batch_size=32, input_size=(68,68,3))
    test_dataloader = DATALOADER.DataLoader(df = df_test, one_hot_encoder = moa_one_hot_encoder, batch_size = 32, input_size=(68,68,3))


    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor = 0.1, patience = 5, min_lr=1e-6)
    check_point = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint , monitor = "val_acc", mode = "max", save_best_only=True)
    csv_logger = tf.keras.callbacks.CSVLogger(path_log, append = True, separator=',')
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-2, momentum=0.9, decay = 0.01),metrics=['accuracy'])

    history=model.fit(train_dataloader, validation_data=test_dataloader, epochs= 50, callbacks = [check_point, rlrop, csv_logger])
    model.summary()
    model.save(path_model)
        
    