from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn import preprocessing 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")
np.random.seed(203)
headers = ["Artist", "Album", "Song", "DateTime"]
lastfm = pd.read_csv('umarsabir.csv', names = headers, parse_dates=[3])

def get_season(row):
    if row['Date'].month >= 4 and row['Date'].month <= 6:
        return 1
    elif row['Date'].month >= 7 and row['Date'].month <= 9:
        return 2
    elif row['Date'].month >= 10 and row['Date'].month <= 12:
        return 3
    else:
        return 4
    
def ampm(row):
    if row["DateTime"].hour < 12:
        return 0
    else:
        return 1
def drop_features(df):
    return df.drop(['DateTime', 'Date', 'Time'], axis=1)
                   
def encode_features(df_train):
    features = ['Artist', 'Album', 'Song', 'Day of the Week', 'Month', 'Day', 'Year', 'Season']
    df_combined = pd.concat([df_train[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
    return df_train

temp = pd.DatetimeIndex(lastfm['DateTime'])
lastfm['DateTime'] = pd.DatetimeIndex(lastfm['DateTime'])
lastfm['Date'] = temp.date
lastfm['Time'] = temp.time

lastfm['Date'] = pd.to_datetime(lastfm['Date'])
lastfm['Day'] = lastfm['Date'].dt.day
DotW = lastfm['Date'].dt.weekday_name
lastfm['Day of the Week'] = DotW
lastfm['Season'] = lastfm.apply(get_season, axis=1)
lastfm["TOD"] = lastfm.apply(ampm, axis=1)
lastfm = lastfm.dropna()
lastfm = lastfm[lastfm.Date.dt.year >= 2012]

lastfm['Month'] = lastfm['Date'].dt.month
lastfm['Year'] = lastfm['Date'].dt.year
df_train = drop_features(lastfm)
df_train = encode_features(df_train)
df_train.head()

x_all = df_train.drop(['TOD'], axis=1)
y_all = df_train['TOD']

num_test = 0.3
X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test, random_state=100)

vc = df_train['TOD'].value_counts().to_frame().reset_index()
vc['percent'] = vc["TOD"].apply(lambda x : round(100*float(x) / len(df_train), 2))
vc = vc.rename(columns = {"index" : "Target", "Class" : "Count"})

non_fraud = df_train[df_train['TOD'] == 0].sample(1000)
fraud = df_train[df_train['TOD'] == 1]

df = non_fraud.append(fraud).sample(frac=1).reset_index(drop=True)
X = df.drop(['TOD'], axis = 1).values
Y = df["TOD"].values

## input layer 
input_layer = Input(shape=(X.shape[1],))

## encoding part
encoded = Dense(100, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoded = Dense(50, activation='relu')(encoded)

## decoding part
decoded = Dense(50, activation='tanh')(encoded)
decoded = Dense(100, activation='tanh')(decoded)

## output layer
output_layer = Dense(X.shape[1], activation='relu')(decoded)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer="adadelta", loss="mse")	

x = df_train.drop(["TOD"], axis=1)
y = df_train["TOD"].values

x_scale = preprocessing.MinMaxScaler().fit_transform(x.values)
x_norm, x_fraud = x_scale[y == 0], x_scale[y == 1]

autoencoder.fit(x_norm[0:2000], x_norm[0:2000], 
                batch_size = 256, epochs = 10, 
                shuffle = True, validation_split = 0.20);

hidden_representation = Sequential()
hidden_representation.add(autoencoder.layers[0])
hidden_representation.add(autoencoder.layers[1])
hidden_representation.add(autoencoder.layers[2])

norm_hid_rep = hidden_representation.predict(x_norm[:3000])
fraud_hid_rep = hidden_representation.predict(x_fraud)
rep_x = np.append(norm_hid_rep, fraud_hid_rep, axis = 0)
y_n = np.zeros(norm_hid_rep.shape[0])
y_f = np.ones(fraud_hid_rep.shape[0])
rep_y = np.append(y_n, y_f)

train_x, val_x, train_y, val_y = train_test_split(rep_x, rep_y, test_size=0.25)
clf = LogisticRegression(solver="lbfgs").fit(train_x, train_y)
pred_y = clf.predict(val_x)

print ("")
print ("Classification Report: ")
print (classification_report(val_y, pred_y))

print ("")
print ("Accuracy Score: ", accuracy_score(val_y, pred_y))