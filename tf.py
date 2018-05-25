import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from input_data import Data
import pandas as pd

data=Data()
X_train,y_train=data.read_data("./data/train.csv",'train')
print(X_train.shape,y_train.shape)

X_test,y_test=data.read_data("./data/test.csv",'test')
print(X_test.shape,y_test.shape)

X=tf.placeholder('float32',[None,X_train.shape[1]])
y=tf.placeholder('float32',[None,])
keep_prob=tf.placeholder('float32')

batch_norm1=slim.layers.batch_norm(X)
dense1=slim.layers.fully_connected(batch_norm1,256)
dense2=slim.layers.fully_connected(dense1,256)
dropout1=slim.layers.dropout(dense2,keep_prob)

# li1=slim.layers.fully_connected(dropout1,64)
# li2=slim.layers.fully_connected(dropout1,64)
# li3=slim.layers.fully_connected(dropout1,64)
# li4=slim.layers.fully_connected(dropout1,64)

li=[]
for i in range(79):
    li1=slim.layers.fully_connected(dropout1,2)
    li.append(li1)

concat1=tf.concat(li,axis=1)
# print(concat1.shape)

dense3=slim.layers.fully_connected(concat1,1)
# flatten1=slim.layers.flatten(dense3)

reshape1=tf.reshape(dense3,(-1,))

print(reshape1.shape)
print(y_train.shape)

loss=slim.losses.absolute_difference(reshape1,y)


# def root_mean_squared_error(y_true, y_pred):
#     return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

loss=tf.sqrt(tf.reduce_mean(tf.square(reshape1-y),axis=-1))

train_op=tf.train.AdamOptimizer(0.001).minimize(loss)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    saver.restore(sess,"./save/model.ckpt")

    # for i in range(15000):
    #     _,loss_val=sess.run([train_op,loss],feed_dict={X:X_train,y:y_train,keep_prob:0.5})
    #     print(i,"=",loss_val)
    #     if i%100 ==99:
    #         saver.save(sess,'./save/model.ckpt')

    y_pred=sess.run(reshape1,feed_dict={X:X_test,keep_prob:1})

    result = pd.DataFrame({"Id": y_test, "SalePrice": y_pred})
    print(result)
    result.to_csv("./sample_submission.csv", index=False)