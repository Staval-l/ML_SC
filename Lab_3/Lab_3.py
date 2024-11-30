from cProfile import label

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.python.keras.saving.save import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm


def task_1():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    plt.show()
    fig, axs = plt.subplots(nrows=1, ncols=10)
    axs[0].imshow(x_train[1, :, :], cmap='gray')
    axs[1].imshow(x_train[3, :, :], cmap='gray')
    axs[2].imshow(x_train[5, :, :], cmap='gray')
    axs[3].imshow(x_train[7, :, :], cmap='gray')
    axs[4].imshow(x_train[9, :, :], cmap='gray')
    axs[5].imshow(x_train[11, :, :], cmap='gray')
    axs[6].imshow(x_train[13, :, :], cmap='gray')
    axs[7].imshow(x_train[15, :, :], cmap='gray')
    axs[8].imshow(x_train[17, :, :], cmap='gray')
    axs[9].imshow(x_train[19, :, :], cmap='gray')
    plt.axis('off')
    plt.show()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return x_train, y_train, x_test, y_test


def create_model(param_1, param_2=0, dropout_val=0.0):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))  # входной слой
    model.add(Dense(param_1, activation='relu', name='hidden_1'))
    if dropout_val != 0:
        model.add(Dropout(dropout_val))
    if param_2 != 0:
        model.add(Dense(param_2, activation='relu', name='hidden_2'))
        if dropout_val != 0:
            model.add(Dropout(dropout_val))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def task_2(x_train, y_train, param_1, param_2=0, dropout_val=0.0):
    # y_train = to_categorical(y_train, num_classes=10)
    # y_test = to_categorical(y_test, num_classes=10)
    validation_split = 0.2
    x_val = x_train[int(len(x_train) * (1 - validation_split)):]
    y_val = y_train[int(len(y_train) * (1 - validation_split)):]
    x_train = x_train[:int(len(x_train) * (1 - validation_split))]
    y_train = y_train[:int(len(y_train) * (1 - validation_split))]

    model = create_model(param_1, param_2, dropout_val)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=16, verbose=0)
    model.save('model.h5')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Обучающая выборка')
    plt.plot(history.history['val_loss'], label='Валидационная выборка')
    plt.title('Функция потерь')
    plt.xlabel('Эпохи')
    plt.ylabel('Потеря')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Обучающая выборка')
    plt.plot(history.history['val_accuracy'], label='Валидационная выборка')
    plt.title('Точность')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()
    plt.show()


def task_3(x_test, y_test, model_path):
    model = load_model(model_path)
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Точность модели: {accuracy:.4f}')
    prediction = model.predict(x_test)
    predicted_classes = np.argmax(prediction, axis=1)
    conf_matrix = confusion_matrix(y_test, predicted_classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.ylabel('Истинные метки')
    plt.xlabel('Предсказанные метки')
    plt.title('Матрица ошибок')
    plt.show()
    prediction_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    print(f'Полученная точность: {prediction_accuracy:.4f}')
    incorrect_indices = np.where(predicted_classes != y_test)[0]
    plt.figure(figsize=(15, 8))
    for i in range(10):
        class_incorrect_indices = incorrect_indices[y_test[incorrect_indices] == i]
        if len(class_incorrect_indices) > 0:
            plt.subplot(2, 5, i + 1)
            plt.imshow(x_test[class_incorrect_indices[0], :, :], cmap='gray')
            plt.title(f'Истинный: {i}, Предсказанный: {predicted_classes[class_incorrect_indices[0]]}')
            plt.axis('off')
    plt.tight_layout()
    plt.show()


def task_4(x_train, y_train):
    model = KerasClassifier(build_fn=create_model, epochs=10, verbose=0)
    param_grid = {
        'param_1': [16, 32, 64, 128],
        'batch_size': [8, 16, 32, 64],
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=2, verbose=0)
    grid_result = grid.fit(x_train, y_train)
    print(f'Лучшие параметры: {grid_result.best_params_}')
    print(f'Лучшее значение точности: {grid_result.best_score_:.4f}')


def task_5(x_train, y_train):
    # 784*x+x+x*10+10
    # 784*x1+x1+x1*x2+x2*10+10

    list_params = [
        [120, 0, 0, 32],
        [120, 0, 0, 64],
        [120, 0, 0.2, 32],
        [120, 0, 0.2, 64],
        [110, 80, 0, 32],
        [110, 80, 0.2, 32],
        [110, 80, 0, 64],
        [110, 80, 0.2, 64]
    ]
    val_loss = []
    val_accuracy = []
    for i in tqdm(range(len(list_params))):
        model = create_model(list_params[i][0], list_params[i][1], list_params[i][2])
        history = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=list_params[i][3], verbose=0)
        val_loss.append(history.history['val_loss'])
        val_accuracy.append(history.history['val_accuracy'])

    plt.figure(figsize=(15, 8))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', '#008080']
    for i in range(len(list_params)):
        plt.plot(val_loss[i], color=colors[i], label=list_params[i])
    plt.title('Функция потерь')
    plt.xlabel('Эпохи')
    plt.ylabel('Потеря')
    plt.legend()
    plt.show()
    for i in range(len(list_params)):
        plt.plot(val_accuracy[i], color=colors[i], label=list_params[i])
    plt.title('Точность')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()
    plt.show()
    print()


def main():
    x_train, y_train, x_test, y_test = task_1()
    task_2(x_train, y_train, 128, 64, 0.2)
    task_3(x_test, y_test, "model.h5")
    task_4(x_train, y_train)
    task_5(x_train, y_train)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
