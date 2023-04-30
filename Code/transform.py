import math
import os
import random
import numpy as np
import cv2
import tensorflow as tf

# Koncepcja: Mamy folder o nazwie piłki, a w tym folderze:
#           * 18 folderów
#           * każdy po 30 zdj
#           * każdy odpowiada jednemu typowi piłki na jednym tle
#           * nazwy jak określiliśmy w docsie, ale bez polskich znaków i z "_" pomiędzy slowami
#               + na końcu 1 słowo okreslające powierzchnię np. pilka_do_noznej_trawa
#           * podział ze względu na to, aby zbiory val i test były zbilansowane pod względem zdjęć
#               z każdej kategorii i z każdego tła


# Funkcja do tworzenia setu walidacyjnego i testowego
def build_subset(image_array, label_array, val_images, val_labels, size):
    random_indices = random.sample(range(len(image_array)), size)

    for i in random_indices:
        val_images.append(image_array[i])
        val_labels.append(label_array[i])

    image_array_removed = [image_array[index] for index in range(len(image_array)) if index not in random_indices]

    label_array = np.array(label_array)
    label_array = np.delete(label_array, random_indices)
    label_array = label_array.tolist()

    return image_array_removed, label_array, val_images, val_labels


# Funkcja aby ujednolicić etykiety na koniec, pozbywając się członu trawa, stol, podloga
def unify_labels(labels_array):
    for i in range(len(labels_array)):
        label_subs = labels_array[i].split('_')
        labels_array[i] = '_'.join(label_subs[:-1])


# Funkcja do normalizacji danych
def normalize_data(img, label):
    return tf.cast(img, tf.float32)/255., label


# pobranie folderów=klas z folderu piłki
directories_classes = []

for root, dirs, _ in os.walk("./pilki", topdown=False):
    for name in dirs:
        directories_classes.append(name)

val_images = []
val_labels = []
test_images = []
test_labels = []
train_images = []
train_labels = []


# pętla po pobranych folderach/klasach
for dir in directories_classes:
    image_array = []
    label_array = []
    # pętla po kolejnych obrazach w danym folderze
    for filename in os.listdir('./pilki/'+dir):
        if filename.endswith('.jpg'):
            img = cv2.imread('./pilki/'+dir+'/'+filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224)) # często taki format w modelach

            # dodaj obraz i etykietę (równą nazwie folderu) do tymczasowych tablic
            image_array.append(img)
            label_array.append(dir)

            # augemntacja danych - flip i zmiana jasności, czyli plus 2x tyle zdjęć
            flipped_img = cv2.flip(img, 1)
            bright_img = tf.image.adjust_brightness(img, delta=-0.4)

            # dodaj nowo postałe dane do tablic
            image_array.append(flipped_img)
            label_array.append(dir)
            image_array.append(bright_img)
            label_array.append(dir)

    # ustaw rozmiar na 10%, zakładając, że podział: 80% - treninigowe, 10% - walidacyjne, 10% - testowe
    size = math.floor(10*len(image_array)/100)
    # wywołaj funkcję losowo przydzielającą do val i test setów
    image_array, label_array, val_images, val_labels = build_subset(image_array, label_array, val_images, val_labels, size)
    image_array, label_array, test_images, test_labels = build_subset(image_array, label_array, test_images, test_labels, size)
    # rozszerz listę train o dane z tymczasowych tablic po usunięciu elementów przydzielonych do val i test
    train_images = train_images + image_array
    train_labels = train_labels + label_array

# zamień listy na numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
val_images = np.array(val_images)
val_labels = np.array(val_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# pozbądź się nazw tła z etykiet
unify_labels(train_labels)
unify_labels(val_labels)
unify_labels(test_labels)

# stwórz datasety train, val, test
dataset_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
dataset_validation = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
dataset_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

# znormalizuj dane
dataset_train = dataset_train.map(normalize_data)
dataset_validation = dataset_validation.map(normalize_data)
dataset_test = dataset_test.map(normalize_data)

# do ustawienia
shuffle_size = 180
dataset_train = dataset_train.shuffle(shuffle_size)
dataset_validation = dataset_validation.shuffle(shuffle_size)
dataset_test = dataset_test.shuffle(shuffle_size)
