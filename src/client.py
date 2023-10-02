import flwr as fl
import tensorflow as tf
import sys
import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import Dense, Input, GlobalMaxPooling1D,Flatten
from keras import backend as K 

port = sys.argv[1]

df_MHE = pd.read_csv('Electricity_MHE.csv')
df_MHE['unix_ts'] = pd.to_datetime(df_MHE['unix_ts'])
df_MHE = df_MHE.set_index('unix_ts')
df_Coustomer = pd.read_csv('Electricity_BME.csv')
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
train_generator = TimeseriesGenerator(in_seq, out_seq, length=n_input, batch_size=30)

input_window_length = 30
nb_features = 1
input_shape=(n_input,1, nb_features)
model_input = Input(shape=input_shape)

test_generator = TimeseriesGenerator(in_seq_test, out_seq_test, length=n_input, batch_size=30)

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

reshape_layer = model_input
conv_layer_1 = tf.keras.layers.Convolution2D(filters=30, kernel_size=(10, 1), strides=(1, 1), padding="same", activation="relu", input_shape=(input_window_length ,1))(reshape_layer)
conv_layer_2 = tf.keras.layers.Convolution2D(filters=30, kernel_size=(8, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_1)
conv_layer_3 = tf.keras.layers.Convolution2D(filters=40, kernel_size=(6, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_2)
conv_layer_4 = tf.keras.layers.Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_3)
conv_layer_5 = tf.keras.layers.Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_4)
flatten_layer = tf.keras.layers.Flatten()(conv_layer_5)
label_layer = tf.keras.layers.Dense(1024, activation="relu")(flatten_layer)
output_layer = tf.keras.layers.Dense(1, activation="linear")(label_layer)

model = tf.keras.Model(inputs=model_input, outputs=output_layer)
adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.99, epsilon=1e-08)
model.compile(loss='mse', optimizer=adam,metrics=['mae',rmse]) 

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(train_generator, epochs=5, verbose=1)
        hist = r.history
        print("Fit history :" , hist)
        #model.save('denoising_autoencoder_windturbine11_2018_feature_extraction_v1_Kalm_Filter_v1_full_feature.h5')
        return model.get_weights(), len(X_train), {}


    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        #loss, mae, rmse = model.evaluate(testSignal, X_test, verbose=0)
        loss, mae, rmse = model.evaluate(test_generator)
        print("Eval loss :", loss)
        print("Eval mae :", mae)
        print("Eval rmse :", rmse)
        #print("Eval accuracy : ", accuracy)
        #return loss, len(testSignal), {"mae": mae}, {"rmse": rmse}
        #return loss, len(testSignal), {"accuracy": accuracy}
        return loss, len(X_test), {"mae": mae}

addr = "sower_platform_container:"+port
fl.client.start_numpy_client(server_address=addr, client=FlowerClient())
