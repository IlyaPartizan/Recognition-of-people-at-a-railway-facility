import cv2
import os
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical
#подключение библиотек


def image_in_np(image_name, path):
    img = np.asarray(Image.open(f'{path}/{image_name}').convert('RGB'))
    return img
# функция преобразования изображения в массив numpy


def finding_the_average_size(images_mas, path_mas):
    all_shapes_x = 0
    count_shapes_x = 0
    all_shapes_y = 0
    count_shapes_y = 0
    for index in range(len(images_mas)):
        for imag in images_mas[index]:
            all_shapes_x += np.shape(image_in_np(imag, path_mas[index]))[0]
            count_shapes_x += 1
            all_shapes_y += np.shape(image_in_np(imag, path_mas[index]))[1]
            count_shapes_y += 1
    average_shape_x = all_shapes_x // count_shapes_x
    average_shape_y = all_shapes_y // count_shapes_y
    return average_shape_x, average_shape_y
#функция вычисления среднего значения shapes X, Y для всех изображений


def resize_and_add_marks_and_matrix(images, path, width, height):
    datas_and_marks = []
    for index in range(len(images)):
        for name in images[index]:
            mark = 0
            img = cv2.imread(f'{path[index]}/{name}')
            new_img = cv2.resize(img, (width, height))
            matrix = np.asarray(new_img)
            if 'bad' in path[index]:
                mark = 1
            elif 'okey' in path[index]:
                mark = 2
            elif 'unknow' in path[index]:
                mark = 3
            elif 'warning' in path[index]:
                mark = 4
            datas_and_marks.append([matrix, mark])
    return datas_and_marks
#функция формирующая набор данных для обучения или тестирования вместе с метками


def decode_output(output):
    for index in range(len(output)):
        output[index] = np.argmax(output[index])
    output = output.tolist()
    new_massive = []
    for index in range(len(output)):
        new_massive.append(int(output[index][0]))
    return new_massive
#функция декодирования выхода нейросети


def check(mas, answer, y):
    new_answer = ''
    if answer == 1:
        new_answer = 'bad'
    if answer == 2:
        new_answer = 'okay'
    if answer == 3:
        new_answer = 'unknown'
    if answer == 4:
        new_answer = 'warning'
    y_new = ''
    if y == 1:
        y_new = 'bad'
    if y == 2:
        y_new = 'okay'
    if y == 3:
        y_new = 'unknown'
    if y == 4:
        y_new = 'warning'

    print("Предсказние:", new_answer)
    print("Метка:", y_new)
    plt.figure()
    plt.imshow(mas)
    plt.colorbar()
    plt.grid(False)
    plt.show()


images_train_bad = os.listdir('train/bad')
images_train_okay = os.listdir('train/okey')
images_train_unknown = os.listdir('train/unknow')
images_train_warning = os.listdir('train/warning')
#формирование списков имен тренировочных данных

images_test_bad = os.listdir('test/bad')
images_test_okay = os.listdir('test/okey')
images_test_unknown = os.listdir('test/unknow')
images_test_warning = os.listdir('test/warning')
#формирование списков имен тестовых данных


all_images_data = [images_train_bad, images_train_okay, images_train_unknown, images_train_warning,
                   images_test_bad, images_test_okay, images_test_unknown, images_test_warning]
#объединение в один массив списков всех имен данных

all_paths = ['train/bad', 'train/okey', 'train/unknow', 'train/warning',
             'test/bad', 'test/okey', 'test/unknow', 'test/warning']
#объединение в один массив списков всех путей до изображений

average_shapes = finding_the_average_size(all_images_data, all_paths)
#нахождение средних значений shapes по X и по Y по всем данным

print("Среднее значение shape по координате X(по всем изображениям:", average_shapes[0],
      "\nСреднее значение shape по координате Y(по всем изображениям:", average_shapes[1])
#вывод средних shapes по X и по Y


train_images_data = [images_train_bad, images_train_okay, images_train_unknown, images_train_warning]
test_images_data = [images_test_bad, images_test_okay, images_test_unknown, images_test_warning]
#подготовка имен тестовых и тренировочных данных для создания тестовой и обучающей выборки с метками

train_paths = ['train/bad', 'train/okey', 'train/unknow', 'train/warning']
test_paths = ['test/bad', 'test/okey', 'test/unknow', 'test/warning']
#подготовка путей тестовых и тренировочных данных для создания тестовой и обучающей выборки с метками


train_data = resize_and_add_marks_and_matrix(train_images_data, train_paths, 100, 100)
test_data = resize_and_add_marks_and_matrix(test_images_data, test_paths, 100, 100)
#создание тестового и обучающего набора данных с метками

train_data = sorted(train_data, key=lambda A: rd.random())
test_data = sorted(test_data, key=lambda A: rd.random())
#перемешивание тестового и обучающего набора данных с метками для лучшего обучения


x_train = [train_data[index][0] for index in range(len(train_data))]
y_train = [train_data[index][1] for index in range(len(train_data))]
x_test = [test_data[index][0] for index in range(len(test_data))]
y_test = [test_data[index][1] for index in range(len(test_data))]
#извлечение меток и данных для нейросети


y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)
#кодирование меток для работы с ними в библиотеке keras


x_train = np.array(x_train)
x_test = np.array(x_test)


model = Sequential()
model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(256, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(5, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(x_train, y_train_encoded, validation_data= (x_test, y_test_encoded), epochs=8)
print(hist.history)
#модель сверточной нейронной сети

plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.title("Точность модели")
plt.ylabel("Точночть")
plt.xlabel("Эпохи")
plt.legend(["учебные", "тестовые"], loc = "upper left")
plt.show()
#вывод графика точности модели

plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("Потери модели")
plt.ylabel("Потери")
plt.xlabel("Эпохи")
plt.legend(["учебные", "тестовые"], loc = "upper left")
plt.show()
#вывод графика потери модели

model.save('model_test.pt')
model.save('model_test.h5')
#сохранение весов модели в форматах pt и h5

model = load_model('model_test.h5')

output_neuronet = model.predict(x_test)
output_neuronet = decode_output(output_neuronet)

count = 0
for index in range(len(y_test)):
    if output_neuronet[index] == y_test[index]:
        count += 1

print("Количество тестовых данных:", len(y_test))
print("Количество правильных предсказаний:", count)


check(x_test.tolist()[0], output_neuronet[0], y_test[0])
check(x_test.tolist()[4], output_neuronet[4], y_test[4])
check(x_test.tolist()[13], output_neuronet[13], y_test[13])
check(x_test.tolist()[14], output_neuronet[14], y_test[14])

names_output_neuronet = ['b', 'o', 'u', 'w']
names_test = ['b', 
              'o', 
              'u', 
              'w']

mas = [[0 for index in range(4)] for index in range(4)]

for index in range(len(y_test)):
    line = y_test[index] - 1
    column = output_neuronet[index] - 1
    mas[line][column] += 1
    
print(names_output_neuronet)
for index in range(len(mas)):
    print(names_test[index], mas[index])
    
train_images_data = [images_train_bad, images_train_okay, images_train_unknown, images_train_warning]
test_images_data = [images_test_bad, images_test_okay, images_test_unknown, images_test_warning]
train_paths = ['train/bad', 'train/okey', 'train/unknow', 'train/warning']
test_paths = ['test/bad', 'test/okey', 'test/unknow', 'test/warning']
accuracy_massive = []
val_accuracy_massive = []
shapes_of_data = []
shape_in_moment = 100

for index in range(15):
    train_data = resize_and_add_marks_and_matrix(train_images_data, train_paths, shape_in_moment, shape_in_moment)
    test_data = resize_and_add_marks_and_matrix(test_images_data, test_paths, shape_in_moment, shape_in_moment)
    train_data = sorted(train_data, key=lambda A: rd.random())
    test_data = sorted(test_data, key=lambda A: rd.random())
    x_train = [train_data[index][0] for index in range(len(train_data))]
    y_train = [train_data[index][1] for index in range(len(train_data))]
    x_test = [test_data[index][0] for index in range(len(test_data))]
    y_test = [test_data[index][1] for index in range(len(test_data))]
    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    model = Sequential()
    model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(shape_in_moment, shape_in_moment, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(x_train, y_train_encoded, validation_data= (x_test, y_test_encoded), epochs=8)
    accuracy_in_moment = hist.history["accuracy"]
    val_accuracy_in_moment = hist.history["val_accuracy"]
    accuracy_massive.append(accuracy_in_moment[len(accuracy_in_moment) - 1])
    val_accuracy_massive.append(val_accuracy_in_moment[len(val_accuracy_in_moment) - 1])
    shapes_of_data.append(shape_in_moment)
    shape_in_moment += 10
    
plt.plot(shapes_of_data, accuracy_massive)
plt.plot(shapes_of_data, val_accuracy_massive)
plt.title("Точность модели в зависимости от размера входной матрицы")
plt.ylabel("Точночть")
plt.xlabel("Размер входной матрицы")
plt.legend(["учебные", "тестовые"], loc = "upper left")
plt.show()
