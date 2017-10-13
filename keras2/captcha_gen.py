#encoding=utf8
'''
导入必要的库
我们需要导入一个叫 captcha 的库来生成验证码。
我们生成验证码的字符由数字和大写字母组成。
'''

from captcha.image import ImageCaptcha

import matplotlib as mpl
# print mpl.rcsetup.interactive_bk # 获取 interactive backend
# print mpl.rcsetup.non_interactive_bk # 获取 non-interactive backend
# print mpl.rcsetup.all_backends # 获取 所有 backend
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import string
characters = string.digits + string.ascii_uppercase
print(characters)

width, height, n_len, n_class = 170, 80, 4, len(characters)
print width, height, n_len, n_class

'''
定义数据生成器
'''
from keras.utils.np_utils import to_categorical

def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y

# def gen_(batch_size=128):
#     X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
#     y = np.zeros((batch_size, n_len), dtype=np.uint8)
#     while True:
#         generator = ImageCaptcha(width=width, height=height)
#         for i in range(batch_size):
#             random_str = ''.join([random.choice(characters) for j in range(4)])
#             X[i] = np.array(generator.generate_image(random_str)).transpose(1, 0, 2)
#             y[i] = [characters.find(x) for x in random_str]
#         yield [X, y, np.ones(batch_size)*int(conv_shape[1]-2), np.ones(batch_size)*n_len], np.ones(batch_size)

'''
测试生成器
'''
def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

if __name__ == '__main__':
    X, y = next(gen(1))
    # import scipy.misc
    # scipy.misc.toimage(decode(y)+'.png', X[0])
    from PIL import Image
    im = Image.fromarray(X[0])
    im.save(decode(y)+'.png')
# X[0].save(decode(y) + '.png')
# plt.imshow(X[0])
# plt.title(decode(y))