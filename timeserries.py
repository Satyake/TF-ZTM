import matplotlib.pyplot as plt

url='https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110-pcp-all-12-1900-2020.json'
import json
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
MMS=MinMaxScaler()
cache_dir='..'
cache_subdir='data'
precip_file=tf.keras.utils.get_file('precip.json',url,extract=True)
precip=[]
time=[]

with open(precip_file,'r') as f:
    precip_raw=json.load(f)
    for i in precip_raw['data'].keys():
        if i=='200001' or i=='199001':
           time.append(i)
        else:
            pass
    for j in precip_raw['data'].values():
        precip.append(j['value'])
#print(precip_raw['data'])
import numpy as np
precip=np.array(precip,dtype=np.float32)
time=np.array(time, dtype=np.int32)
#print(time)
#print(precip)
#normalize
precip_std=MMS.fit_transform(precip.reshape(-1,1))
#print(precip_std)

split_size=int(0.8*len(precip_std))
train_precip=precip_std[:split_size]
train_time=precip_std[:split_size]
test_precip=precip_std[split_size:]
test_time=precip_std[split_size:]
#print(len(train_precip)),print(len(train_time)), print(len(test_precip)),print(len( test_time))

#windowing the data
train_ds=tf.data.Dataset.from_tensor_slices((train_precip))
train_time_ds=tf.data.Dataset.from_tensor_slices((train_time))

train_ds=train_ds.window(5,shift=1,drop_remainder=True).flat_map(lambda b:b.batch(5))
#train_time_ds=train_time_ds.window(5,shift=1, drop_remainder=True).flat_map(lambda b:b.batch(5))
train_ds=train_ds.map(lambda x: (x[:-1],x[-1]))

#train_dataset=tf.data.Dataset.zip((train_ds,train_time_ds))

test_ds=tf.data.Dataset.from_tensor_slices((test_precip))
#test_time_ds=tf.data.Dataset.from_tensor_slices((test_time))

test_ds=test_ds.window(5,shift=1,drop_remainder=True).flat_map(lambda b:b.batch(5))
test_ds=test_ds.map(lambda x:(x[:-1],x[-1]))
#test_time_ds=test_time_ds.window(5,shift=1, drop_remainder=True).flat_map(lambda b:b.batch(5))
#test_dataset=tf.data.Dataset.zip((test_ds,test_time_ds))

train_ds,test_ds=train_ds.batch(32),test_ds.batch(32)

for d in train_ds:
    print(d[1])
model=tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(None,1)))
model.add(tf.keras.layers.LSTM(56, activation='relu', return_sequences=True))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100,activation='relu')))
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer='rmsprop', metrics=['mae'], loss='mae')
#model.fit(train_ds,validation_data=test_ds, epochs=100)
print(model.summary())
#preds=model.predict(test_precip)
#print(preds.shape)
#preds=tf.squeeze(preds,axis=-1)
#print(preds.shape)
#preds=tf.squeeze(preds,axis=-1)
plt.plot(preds)
plt.plot(test_precip)
plt.show()

#model.evaluate(test_ds)


