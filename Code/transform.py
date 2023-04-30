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


def flip_image(img, direction):
    return cv2.flip(img, direction)


def adjust_brightness(img, delta):
    return tf.image.adjust_brightness(img, delta=delta)


def rotate_image(img, angle_range_left, angle_range_right):
    angle = np.random.randint(angle_range_left, angle_range_right)
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def zoom_image(img, zoom_factor):
    h, w = img.shape[:2]
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    center_x, center_y = int(w / 2), int(h / 2)
    cropped_img = img[center_y - new_h:center_y + new_h, center_x - new_w:center_x + new_w]
    return cv2.resize(cropped_img, (w, h))


# pobranie folderów=klas z folderu piłki
def main():
    directories_classes = []

    for root, dirs, _ in os.walk("./pilki", topdown=False):
        for name in dirs:
            directories_classes.append(name)

    val_images_split1 = []
    val_labels_split1 = []
    val_images_split2 = []
    val_labels_split2 = []

    test_images_split1 = []
    test_labels_split1 = []
    test_images_split2 = []
    test_labels_split2 = []
    test_images_split3 = []
    test_labels_split3 = []

    train_images_split1 = []
    train_labels_split1 = []
    train_images_split2 = []
    train_labels_split2 = []
    train_images_split3 = []
    train_labels_split3 = []


    # pętla po pobranych folderach/klasach
    for dir in directories_classes:
        image_array_split1 = []
        label_array_split1 = []
        image_array_split2 = []
        label_array_split2 = []
        image_array_split3 = []
        label_array_split3 = []
        # pętla po kolejnych obrazach w danym folderze
        for filename in os.listdir('./pilki/'+dir):
            if filename.endswith('.jpg'):
                img = cv2.imread('./pilki/'+dir+'/'+filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224)) # często taki format w modelach

                # dodaj obraz i etykietę (równą nazwie folderu) do tymczasowych tablic
                image_array_split1.append(img)
                label_array_split1.append(dir)
                image_array_split2.append(img)
                label_array_split2.append(dir)
                image_array_split3.append(img)
                label_array_split3.append(dir)

                #augemntacja danych - plus 4x tyle zdjęć
                #flip pionowy względem osi x
                flipped_img = flip_image(img, 0)
                #zmiana jasności (przyciemnienie)
                bright_img = adjust_brightness(img, -0.4)
                #obrót obrazu o losowy kąt z zakresu (-90, 90) stopni
                rotated_img = rotate_image(img, -90, 90)
                #zoom x 2
                zoomed_img = zoom_image(img, 2)

                # dodaj nowo postałe dane do tablic
                image_array_split2.append(flipped_img)
                label_array_split2.append(dir)
                image_array_split2.append(bright_img)
                label_array_split2.append(dir)
                image_array_split2.append(rotated_img)
                label_array_split2.append(dir)
                image_array_split2.append(zoomed_img)
                label_array_split2.append(dir)

                image_array_split3.append(flipped_img)
                label_array_split3.append(dir)
                image_array_split3.append(bright_img)
                label_array_split3.append(dir)
                image_array_split3.append(rotated_img)
                label_array_split3.append(dir)
                image_array_split3.append(zoomed_img)
                label_array_split3.append(dir)

        # ustaw rozmiar na 10%, zakładając, że podział: 80% - treninigowe, 10% - walidacyjne, 10% - testowe
        size_split1 = math.floor(10*len(image_array_split1)/100)
        size_split2_3 = math.floor(10 * len(image_array_split2) / 100)

        # wywołaj funkcję losowo przydzielającą do val i test setów
        image_array_split1, label_array_split1, test_images_split1, test_labels_split1 = build_subset(image_array_split1, label_array_split1, test_images_split1, test_labels_split1, size_split1)
        image_array_split1, label_array_split1, val_images_split1, val_labels_split1 = build_subset(image_array_split1,label_array_split1,val_images_split1,val_labels_split1,size_split1)

        image_array_split2, label_array_split2, test_images_split2, test_labels_split2 = build_subset(image_array_split2, label_array_split2, test_images_split2,test_labels_split2, size_split2_3)
        image_array_split2, label_array_split2, val_images_split2, val_labels_split2 = build_subset(image_array_split2,label_array_split2,val_images_split2,val_labels_split2,size_split2_3)

        image_array_split3, label_array_split3, test_images_split3, test_labels_split3 = build_subset(image_array_split3, label_array_split3, test_images_split3,test_labels_split3, size_split2_3)

        # rozszerz listę train o dane z tymczasowych tablic po usunięciu elementów przydzielonych do val i test
        train_images_split1 = train_images_split1 + image_array_split1
        train_labels_split1 = train_labels_split1 + label_array_split1

        train_images_split2 = train_images_split2 + image_array_split2
        train_labels_split2 = train_labels_split2 + label_array_split2

        train_images_split3 = train_images_split3 + image_array_split3
        train_labels_split3 = train_labels_split3 + label_array_split3

    # zamień listy na numpy arrays
    train_images_split1 = np.array(train_images_split1)
    train_labels_split1 = np.array(train_labels_split1)
    val_images_split1 = np.array(val_images_split1)
    val_labels_split1 = np.array(val_labels_split1)
    test_images_split1 = np.array(test_images_split1)
    test_labels_split1 = np.array(test_labels_split1)

    train_images_split2 = np.array(train_images_split2)
    train_labels_split2 = np.array(train_labels_split2)
    val_images_split2 = np.array(val_images_split2)
    val_labels_split2 = np.array(val_labels_split2)
    test_images_split2 = np.array(test_images_split2)
    test_labels_split2 = np.array(test_labels_split2)

    train_images_split3 = np.array(train_images_split3)
    train_labels_split3 = np.array(train_labels_split3)
    test_images_split3 = np.array(test_images_split3)
    test_labels_split3 = np.array(test_labels_split3)

    # pozbądź się nazw tła z etykiet
    # unify_labels(train_labels)
    # unify_labels(val_labels)
    # unify_labels(test_labels)

    # stwórz datasety train, val, test
    dataset_train_split1 = tf.data.Dataset.from_tensor_slices((train_images_split1, train_labels_split1))
    dataset_validation_split1 = tf.data.Dataset.from_tensor_slices((val_images_split1, val_labels_split1))
    dataset_test_split1 = tf.data.Dataset.from_tensor_slices((test_images_split1, test_labels_split1))

    dataset_train_split2 = tf.data.Dataset.from_tensor_slices((train_images_split2, train_labels_split2))
    dataset_validation_split2 = tf.data.Dataset.from_tensor_slices((val_images_split2, val_labels_split2))
    dataset_test_split2 = tf.data.Dataset.from_tensor_slices((test_images_split2, test_labels_split2))

    dataset_train_split3 = tf.data.Dataset.from_tensor_slices((train_images_split3, train_labels_split3))
    dataset_test_split3 = tf.data.Dataset.from_tensor_slices((test_images_split3, test_labels_split3))

    # znormalizuj dane
    dataset_train_split2 = dataset_train_split2.map(normalize_data)
    dataset_validation_split2 = dataset_validation_split2.map(normalize_data)
    dataset_test_split2 = dataset_test_split2.map(normalize_data)

    dataset_train_split3 = dataset_train_split3.map(normalize_data)
    dataset_test_split3 = dataset_test_split3.map(normalize_data)

    # do ustawienia
    shuffle_size = 180
    dataset_train_split1 = dataset_train_split1.shuffle(shuffle_size)
    dataset_validation_split1 = dataset_validation_split1.shuffle(shuffle_size)
    dataset_test_split1 = dataset_test_split1.shuffle(shuffle_size)

    dataset_train_split2 = dataset_train_split2.shuffle(shuffle_size)
    dataset_validation_split2 = dataset_validation_split2.shuffle(shuffle_size)
    dataset_test_split2 = dataset_test_split2.shuffle(shuffle_size)

    dataset_train_split3 = dataset_train_split3.shuffle(shuffle_size)
    dataset_test_split3 = dataset_test_split3.shuffle(shuffle_size)

    print(len(train_labels_split1))
    print(len(val_labels_split1))
    print(len(test_labels_split1))

    print(len(dataset_train_split1))
    print(len(val_labels_split1))
    print(len(dataset_test_split1))

    print(len(train_labels_split2))
    print(len(val_labels_split2))
    print(len(test_labels_split2))

    print(len(dataset_train_split2))
    print(len(val_labels_split2))
    print(len(dataset_test_split2))

    print(len(train_labels_split3))
    print(len(test_labels_split3))

    print(len(dataset_train_split3))
    print(len(dataset_test_split3))


main()
