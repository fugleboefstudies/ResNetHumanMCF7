import tensorflow as tf

def init_default_model(model_number):
    model = tf.keras.models.Sequential()
    
    #Data Augmentation
    model.add(tf.keras.layers.RandomFlip())
    model.add(tf.keras.layers.RandomRotation(0.1))
    model.add(tf.keras.layers.Conv2D(32,3,strides=2,padding="same"))
    model.add(tf.keras.layers.Conv2D(32,3,strides=2,padding="same"))
    model.add(tf.keras.layers.MaxPooling2D(3, strides=2, padding="same"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())
    
    if model_number == 0:
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        
    elif model_number == 1:
        model.add(tf.keras.layers.Flatten()) 
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        for i in range(4):
            model.add(tf.keras.layers.Dense(64,activation='relu'))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.BatchNormalization())
            
    elif model_number == 2:
        model.add(tf.keras.layers.Flatten()) 
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        for i in range(4):
            model.add(tf.keras.layers.Dense(128,activation='relu'))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.BatchNormalization())
        for i in range(5):
            model.add(tf.keras.layers.Dense(64,activation='relu'))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.BatchNormalization())
            
    elif model_number == 3:
        model.add(tf.keras.layers.Flatten()) 
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        for i in range(4):
            model.add(tf.keras.layers.Dense(128,activation='relu'))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.BatchNormalization())
        for i in range(5):
            model.add(tf.keras.layers.Dense(64,activation='relu'))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.BatchNormalization())
        for i in range(5):
            model.add(tf.keras.layers.Dense(32,activation='relu'))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(12, activation = 'softmax'))
    
    
    return model
