import numpy as np
import pandas as pd
import tensorflow as tf


class DataLoader(tf.keras.utils.Sequence):
    #def __init__(self, df, model = None, one_hot_encoder, learning_rate_schedule = None, batch_size, input_size = (68,68,3), shuffle = True):
    def __init__(self, df, one_hot_encoder, batch_size, model = None, input_size = (68,68,3), shuffle = True):
        self.df = df
        self.dmsomean = np.load('F:\Programming\DTU\Human MCF7\Segmented\Inspection\ClassMean\DMSO.npy')
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        
        self.n = len(self.df)
        self.n_classes = self.df['moa'].nunique()
        
        self.one_hot_encoder = one_hot_encoder
        
        #self.learning_rate_schedule = iter(learning_rate_schedule)
        
        self.on_epoch_end()
        
        
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            #self.df = self.df.groupby('moa').sample(n=3000, random_state=0)
        
        # lr = self.get_next_learning_rate()
        # if lr:
        #     tf.keras.backend.set_value(model.optimizer.learning_rate, lr)
        #     print(f"Learning rate adjusted to: {lr}")
            
    # def get_next_learning_rate(self):
    #     try:
    #         lr = next(self.learning_rate_schedule)
    #     except StopIteration:
    #         lr = None
    #     return lr

    def __get_img(self, path):
        image_arr = np.load(path)
        image_arr = image_arr/255
        image_arr -= self.dmsomean #Normalize by dmso

        image_arr = tf.image.resize(image_arr, (self.input_size[0], self.input_size[1]))
        return image_arr
    
    def __get_label(self, moa):
        #print(moa)
        label = self.one_hot_encoder.transform(moa.to_numpy().reshape(-1, 1))
        return label
    
    def __get_batch(self, batch):

        img_batch = batch['path'].apply(self.__get_img)
        img_batch = np.array([img for img in img_batch])
        img_batch = tf.keras.applications.resnet50.preprocess_input(img_batch)
        

        #label_batch = batch['moa'].apply(self.__get_label)
        label_batch = self.one_hot_encoder.transform(batch.moa.to_numpy().reshape(-1, 1))
        
        return img_batch, label_batch
    
    def __getitem__(self, index):
        batch = self.df[index * self.batch_size:(index+1)*self.batch_size]
        X, Y = self.__get_batch(batch)
        return X, Y
        
    def __len__(self):
        return self.n // self.batch_size