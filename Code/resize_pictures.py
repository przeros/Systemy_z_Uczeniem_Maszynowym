import cv2
import numpy as np
import os

# do resizowania zdjęć na ratio 4:3
# z folderu input, ustawionego tu na to_be_resized bierze zdjęcia, zmienia ich rozmiar
# i zmienione wrzuca do folderu output - resized

def resize(input_path, output_path):

    for filename in os.listdir(input_path):
            if filename.endswith('.jpg'):
                image_path = os.path.join(input_path, filename)
                img = cv2.imread(image_path)
                height, width, _ = img.shape
                current_ratio = width / height
                wanted_ratio = 4 / 3

            if current_ratio > wanted_ratio:
                # image is wider than 4:3 - crop the sides
                new_width = int(height * wanted_ratio)
                left = (width - new_width) // 2
                right = width - new_width - left
                img = img[:, left:-right, :]
            else:
                # taller than 4:3- add borders at the top and bottom
                new_height = int(width / wanted_ratio)
                top = (new_height - height) // 2
                bottom = new_height - height - top
                img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)

            img = cv2.resize(img, (4000, 3000))

            cv2.imwrite(os.path.join(output_path, filename), img)

input_path = './to_be_resized'
output_path = './resized'

resize(input_path, output_path)