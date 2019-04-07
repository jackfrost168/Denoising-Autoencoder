import numpy as np
import tensorflow as tf
#print("TensorFlow Version: %s" % tf.__version__)

import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
# 读取数据集，第一次TensorFlow会自动下载数据集到下面的路径中, label 采用 one_hot 形式
# label 默认采用 0~9 来表示，等价于 one_hot=False, read_data_sets 时会默认把图像 reshape(展平)
# 若想保留图像的二维结构，可以传入 reshape=False
mnist = input_data.read_data_sets('MNIST_data', validation_size=0, one_hot=False)
#img = mnist.train.images[10]
#plt.imshow(img.reshape((28, 28)))
#plt.show()

inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs_')
targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets_')

conv1 = tf.layers.conv2d(inputs_, 64, (3,3), padding='same', activation=tf.nn.relu)
                        #（input,深度，3*3，0填充）strides默认（1，1）
conv1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')
                                #(input, 过滤器尺寸， 步长x和y方向，零填充)

conv2 = tf.layers.conv2d(conv1, 64, (3,3), padding='same', activation=tf.nn.relu)
conv2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')

conv3 = tf.layers.conv2d(conv2, 32, (3,3), padding='same', activation=tf.nn.relu)
conv3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')

#重新改变图像尺寸，使用双线性插值法 https://blog.csdn.net/zsean/article/details/76383100
conv4 = tf.image.resize_nearest_neighbor(conv3, (7,7))
conv4 = tf.layers.conv2d(conv4, 32, (3,3), padding='same', activation=tf.nn.relu)

conv5 = tf.image.resize_nearest_neighbor(conv4, (14,14))
conv5 = tf.layers.conv2d(conv5, 64, (3,3), padding='same', activation=tf.nn.relu)

conv6 = tf.image.resize_nearest_neighbor(conv5, (28,28))
conv6 = tf.layers.conv2d(conv6, 64, (3,3), padding='same', activation=tf.nn.relu)

logits_ = tf.layers.conv2d(conv6, 1, (3,3), padding='same', activation=None)

outputs_ = tf.nn.sigmoid(logits_, name='outputs_')  #激活函数：y = 1/(1 + exp (-x))

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits_)#sigmoid函数交叉熵
#它对于输入的logits先通过sigmoid函数计算，再计算它们的交叉熵，但是它对交叉熵的计算方式进行了优化，使得结果不至于溢出
#output不是一个数，而是一个batch中每个样本的loss,所以一般配合tf.reduce_mean(loss)使用
#https://blog.csdn.net/QW_sunny/article/details/72885403
cost = tf.reduce_mean(loss) #计算均值

optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)    #Adam优化

sess = tf.Session()

noise_factor = 0.5
epochs = 10
batch_size = 128
sess.run(tf.global_variables_initializer()) #初始化模型参数
#https://blog.csdn.net/u012436149/article/details/78291545
#=======train=============
i=0
for e in range(epochs):
    for idx in range(mnist.train.num_examples // batch_size):
        batch = mnist.train.next_batch(batch_size)  #随机取出 batch_size 个图片及其类标
        imgs = batch[0].reshape((-1, 28, 28, 1))    #-1表示输入图片数量不确定

        # 加入噪声
        noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)#通过本函数可以返回一个或一组服从标准正态分布的随机样本值。
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)    #将noisy_imgs限制在[0，1]
        batch_cost, _ = sess.run([cost, optimizer],
                                 feed_dict={inputs_: noisy_imgs,
                                            targets_: imgs})

        print("Epoch: {}/{} ".format(e + 1, epochs),    #字符串格式化，format 函数可以接受不限个参数，位置可以不按顺序。
              "Training loss: {:.4f}".format(batch_cost))
        plt.scatter(i,batch_cost)
        i=i+1
plt.show()

#======test===============
fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(20,4))
#fig整个图，axes坐标轴和绘制的图
in_imgs = mnist.test.images[10:20]
noisy_imgs = in_imgs + noise_factor * np.random.randn(*in_imgs.shape)
noisy_imgs = np.clip(noisy_imgs, 0., 1.)

reconstructed = sess.run(outputs_,
                         feed_dict={inputs_: noisy_imgs.reshape((10, 28, 28, 1))})

for images, row in zip([in_imgs, noisy_imgs, reconstructed], axes):#zip(a,b)将a，b打包成元组
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)))
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)

plt.savefig("Aecresults.png")
fig.show()
plt.draw()
print(i)
plt.waitforbuttonpress()
fig.tight_layout(pad=0.1)
sess.close()

