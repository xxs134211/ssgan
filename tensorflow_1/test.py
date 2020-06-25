import tensorflow as tf
import numpy as np

Truth = np.array([0, 0, 1, 0])
Pred_logits = np.array([3.5, 2.1, 7.89, 4.4])

loss = tf.nn.softmax_cross_entropy_with_logits(labels=Truth, logits=Pred_logits)
loss2 = tf.nn.softmax_cross_entropy_with_logits(labels=Truth, logits=Pred_logits)
loss3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(Truth), logits=Pred_logits)


print(loss)
print(loss2)
print(loss3)
