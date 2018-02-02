from captcha.image import ImageCaptcha
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import random
from keras.models import *
from keras.layers import *

import string

BATCH_SIZE = 128
IMAGE_WIDTH = 170
IMAGE_HEIGHT = 80
n_class = 36
n_len = 4
characters = string.digits + string.ascii_uppercase
project_root_path = os.path.abspath('..')
images_dir = project_root_path + "/image/"
image_num = 100
epochs = 10
model_name = project_root_path + "/model/keras_captcha.model"


def get_randm_image(target_str, random_flag):
    width, height, n_len, n_class = 170, 80, 4, len(characters)

    generator = ImageCaptcha(width=width, height=height)
    if not random_flag:
        random_str = target_str
    else:
        random_str = ''.join([random.choice(characters) for j in range(n_len)])
    img = generator.generate_image(random_str)

    plt.imshow(img)
    plt.title(random_str)
    plt.show()
    return img


def get_randm_images_batches(image_path, image_num):
    global IMAGE_WIDTH, IMAGE_HEIGHT
    for image_index in range(image_num):
        img = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
        random_str = ''.join([random.choice(characters) for j in range(n_len)])
        img.write(random_str, image_path + random_str + '.png')


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y])


def gen(batch_size=32):
    X = np.zeros((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y


def get_data(image_file_dir=project_root_path + "/image"):
    global IMAGE_WIDTH, IMAGE_HEIGHT, n_class, n_len
    file_name_list, file_str_list = get_filename_list(image_file_dir)
    sample_num = len(file_name_list)
    X = np.zeros((sample_num, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    y = [np.zeros((sample_num, n_class), dtype=np.uint8) for i in range(n_len)]
    for i in range(sample_num):
        file_path = file_name_list[i]
        file_str = file_str_list[i]
        image_file = mpimg.imread(file_path)
        image_arr = image_file.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        X[i] = image_arr
        for j, ch in enumerate(file_str):
            y[j][i, :] = 0
            y[j][i, characters.find(ch)] = 1
    return X, y


def build_network(input_image_height, input_image_width):
    input_tensor = Input((input_image_height, input_image_width, 3))
    x = input_tensor
    for i in range(4):
        x = Convolution2D(32 * 2 ** i, 3, 3, activation='relu')(x)
        x = Convolution2D(32 * 2 ** i, 3, 3, activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = [Dense(n_class, activation='softmax', name='c%d' % (i + 1))(x) for i in range(4)]
    model = Model(input=input_tensor, output=x)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


def load_trained_model(path, input_image_height, input_image_width):
    model = build_network(input_image_height, input_image_width)
    model.load_weights(path)
    return model


def get_filename_list(pos_file_dir):
    pos_dir_file_list = os.listdir(pos_file_dir)  # 列出文件夹下所有的目录与文件
    pos_file_name_list = []
    pos_file_str_list = []
    print("========Load Files==========")
    print("Files in dir " + pos_file_dir)
    for i in range(0, len(pos_dir_file_list)):
        path = os.path.join(pos_file_dir, pos_dir_file_list[i])
        # path = unicode(path , "GB2312")
        print(path + ", " + pos_dir_file_list[i])
        if os.path.isfile(path) and not pos_dir_file_list[i].startswith("."):
            pos_file_name_list.append(path)
            pos_file_str_list.append(pos_dir_file_list[i].replace(".png", ""))
        else:
            print("File exception:" + path)
    return pos_file_name_list, pos_file_str_list


def learning_curve(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


if __name__ == "__main__":
    # Step 1:Generate the captcha images
    # get_randm_images_batches(images_dir, image_num)
    # Step 2:Train, evaluate and save the model on the given dataset for the first time
    X, y = get_data(images_dir)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    X_train, X_test, y_train, y_test = X, X, y, y
    model = build_network(IMAGE_HEIGHT, IMAGE_WIDTH)
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=epochs, validation_split=0.33)
    score = model.evaluate(X_test, y_test)
    print(score)
    learning_curve(history)
    model.save(model_name)
    # Step 3:Load the saved model and test it with single sample
    model = load_trained_model(model_name, IMAGE_HEIGHT, IMAGE_WIDTH)
    random_index = np.random.randint(0, X.shape[0])
    xtest, ytest = X[random_index], y[random_index]
    y_pred = model.predict(xtest)
    plt.title('real: %s\npred:%s' % (decode(ytest), decode(y_pred)))
    plt.imshow(xtest[0], cmap='gray')
    plt.show()
