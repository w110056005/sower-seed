import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
import flwr as fl

from keras.layers import Dense, Input
from keras.layers import Conv1D, GRU,Bidirectional

from keras import backend as K
from keras.preprocessing.sequence import TimeseriesGenerator

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.options.display.max_rows=50
np.random.seed(1337)

keras.backend.set_epsilon(1)
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.07
session = tf.compat.v1.Session(config=config) 

df_MHE = pd.read_csv('dataset/Electricity_MHE.csv')
df_MHE['unix_ts'] = pd.to_datetime(df_MHE['unix_ts'])
df_MHE = df_MHE.set_index('unix_ts')
df_Coustomer = pd.read_csv('dataset/Electricity_BME.csv')
df_Coustomer['unix_ts'] = pd.to_datetime(df_Coustomer['unix_ts'])
df_Coustomer = df_Coustomer.set_index('unix_ts')
df_MHE_2013 = df_MHE[395580:]
aggregate_df = df_MHE_2013[:-130019]
df_Coustomer_2013 = df_Coustomer[395580:]
app = df_Coustomer_2013[:-130019]
aggregate_df = aggregate_df.drop(['V','I','f', 'DPF', 'APF', 'Pt', 'Q', 'Qt', 'S','St'], axis=1)
app = app.drop(['V','I','f', 'DPF', 'APF', 'Pt', 'Q', 'Qt', 'S','St'], axis=1)
training_size = int(len(aggregate_df)*0.8)
X_train = aggregate_df[:training_size]
y_train = app[:training_size]
X_test = aggregate_df[training_size:]
y_test = app[training_size:]
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

in_seq = X_train.reshape((len(X_train), 1))
out_seq = y_train.reshape((len(y_train), 1))

in_seq_test = X_test.reshape((len(X_test), 1))
out_seq_test = y_test.reshape((len(y_test), 1))

n_input = 1
nb_out = 1

train_generator = TimeseriesGenerator(in_seq, out_seq, length=n_input, batch_size=60)

nb_features = 1
input_shape=(n_input, nb_features)
model_input = Input(shape=input_shape)

test_generator = TimeseriesGenerator(in_seq_test, out_seq_test, length=n_input, batch_size=60)

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

x = Conv1D(16, 4, activation="relu", padding="same", strides=1, input_shape=input_shape)(model_input)
x = (Conv1D(8, 4, activation="relu", padding="same", strides=1))(x)
x = (Bidirectional(GRU(64, return_sequences=True, stateful=False), merge_mode='concat'))(x)
x = (Bidirectional(GRU(128, return_sequences=False, stateful=False), merge_mode='concat'))(x)
x = (Dense(64, activation='relu'))(x)
x = (Dense(1, activation='linear'))(x)
model = keras.Model(model_input, x)
adam = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mse', optimizer=adam,metrics=['mae',rmse]) 

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit_generator(train_generator, epochs=5, verbose=1)
        hist = r.history
        print("Fit history :" , hist)
        #model.save('denoising_autoencoder_windturbine11_2018_feature_extraction_v1_Kalm_Filter_v1_full_feature.h5')
        return model.get_weights(), len(X_train), {}


    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        #loss, mae, rmse = model.evaluate(testSignal, X_test, verbose=0)
        loss, mae, rmse = model.evaluate_generator(test_generator)
        print("Eval loss :", loss)
        print("Eval mae :", mae)
        print("Eval rmse :", rmse)
        #print("Eval accuracy : ", accuracy)
        #return loss, len(testSignal), {"mae": mae}, {"rmse": rmse}
        #return loss, len(testSignal), {"accuracy": accuracy}
        return loss, len(X_test), {"mae": mae}

client = FlowerClient()
fl.client.start_numpy_client(server_address="sower_platform_container:8080", client=client)