import tensorflow as tf

def init_default_model(model_number):
    model = tf.keras.models.Sequential()
    # if model_number == 0:
    #     model.add(tf.keras.layers.Flatten()) 
    #     model.add(tf.keras.layers.Dense(64, input_dim = 13872, activation='relu'))
        
    # elif model_number == 1:
    #     model.add(tf.keras.layers.Flatten()) 
    #     model.add(tf.keras.layers.Dense(64, input_dim = 13872, activation='relu'))
    #     for i in range(4):
    #         model.add(tf.keras.layers.Dense(64,activation='relu'))
    #         model.add(tf.keras.layers.Dropout(0.2))
    #         model.add(tf.keras.layers.BatchNormalization())
            
    # elif model_number == 2:
    #     model.add(tf.keras.layers.Flatten()) 
    #     model.add(tf.keras.layers.Dense(128, input_dim = 13872, activation='relu'))
    #     for i in range(4):
    #         model.add(tf.keras.layers.Dense(128,activation='relu'))
    #         model.add(tf.keras.layers.Dropout(0.2))
    #         model.add(tf.keras.layers.BatchNormalization())
    #     for i in range(5):
    #         model.add(tf.keras.layers.Dense(64,activation='relu'))
    #         model.add(tf.keras.layers.Dropout(0.2))
    #         model.add(tf.keras.layers.BatchNormalization())
            
    # elif model_number == 3:
    #     model.add(tf.keras.layers.Flatten()) 
    #     model.add(tf.keras.layers.Dense(128, input_dim = 13872, activation='relu'))
    #     for i in range(4):
    #         model.add(tf.keras.layers.Dense(128,activation='relu'))
    #         model.add(tf.keras.layers.Dropout(0.2))
    #         model.add(tf.keras.layers.BatchNormalization())
    #     for i in range(5):
    #         model.add(tf.keras.layers.Dense(64,activation='relu'))
    #         model.add(tf.keras.layers.Dropout(0.2))
    #         model.add(tf.keras.layers.BatchNormalization())
    #     for i in range(5):
    #         model.add(tf.keras.layers.Dense(32,activation='relu'))
    #         model.add(tf.keras.layers.Dropout(0.2))
    #         model.add(tf.keras.layers.BatchNormalization())
    
    model.add(tf.keras.layers.Flatten()) 
    model.add(tf.keras.layers.Dense(128, input_dim = 13872, activation='relu'))
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

def init_model(modelnumber):
    input_tensor = tf.keras.Input(shape=(68,68,3))
    model_res = tf.keras.applications.ResNet50V2(include_top = False, weights = None, input_tensor = input_tensor)
    model = tf.keras.models.Sequential()

    model.add(model_res) #Resnet
    model.add(tf.keras.layers.Flatten()) 
    
    if modelnumber == 0:
        pass
    elif modelnumber == 1:
        model.add(tf.keras.layers.Dense(64,activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())
    elif modelnumber == 2:
        model.add(tf.keras.layers.Dense(64,activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(64,activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())
        
    elif modelnumber == 3:
        model.add(tf.keras.layers.Dense(128,activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(64,activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(32,activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())


    model.add(tf.keras.layers.Dense(12, activation = 'softmax'))
    
    
    return model