#encoding=utf8

from captcha_gen import *
'''
定义网络结构
'''
from keras.models import *
from keras.layers import *

input_tensor = Input((height, width, 3))
x = input_tensor
for i in range(4):
    # x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    # x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = Conv2D(32*2**i, (3, 3), activation='relu')(x) # keras 2.x
    x = Conv2D(32*2**i, (3, 3), activation='relu')(x) # keras 2.x
    x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
# model = Model(input=input_tensor, output=x)
model = Model(inputs=input_tensor, outputs=x) # keras 2.x

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

'''
网络结构可视化
'''
# from keras.utils.visualize_util import plot # keras 1.x
from keras.utils.vis_utils import plot_model as plot # keras 2.x
from IPython.display import Image

plot(model, to_file="model.png", show_shapes=True)
Image('model.png')



'''
计算模型总体准确率
'''
from tqdm import tqdm
def evaluate(model, batch_num=20):
    batch_acc = 0
    generator = gen()
    for i in tqdm(range(batch_num)):
        X, y = generator.next()
        y_pred = model.predict(X)
        batch_acc += np.mean(map(np.array_equal, np.argmax(y, axis=2).T, np.argmax(y_pred, axis=2).T))
    return batch_acc / batch_num

'''
训练模型
'''
# model.fit_generator(gen(), samples_per_epoch=51200, nb_epoch=5,
#                     validation_data=gen(), nb_val_samples=1280)
for i in range(51200 * 5 / 1024):
    # x, y = next(gen())
    # model.train_on_batch(x, y)
    model.fit_generator(gen(), samples_per_epoch=256, nb_epoch=5, validation_data=gen(), nb_val_samples=1280)
    print i, evaluate(model)
    model.save('cnn_%d.h5' % i)


'''
测试模型
'''
X, y = next(gen(1))
y_pred = model.predict(X)
# plt.title('real: %s\npred:%s'%(decode(y), decode(y_pred)))
# plt.imshow(X[0], cmap='gray')
# plt.axis('off')
from PIL import Image
im = Image.fromarray(X[0])
im.save(decode(y)+'_'+decode(y_pred)+'.png')


# '''
# 计算模型总体准确率
# '''
# from tqdm import tqdm
# def evaluate(model, batch_num=20):
#     batch_acc = 0
#     generator = gen()
#     for i in tqdm(range(batch_num)):
#         X, y = generator.next()
#         y_pred = model.predict(X)
#         batch_acc += np.mean(map(np.array_equal, np.argmax(y, axis=2).T, np.argmax(y_pred, axis=2).T))
#     return batch_acc / batch_num

# evaluate(model)

'''
保存模型
'''
model.save('cnn.h5')




