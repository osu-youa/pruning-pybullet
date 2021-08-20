import numpy as np
import os
import PIL
from PIL import Image


def process_grayscale_image(img_array):

    if len(img_array.shape) == 3:
        img_array = img_array.mean(axis=2)
    unwrapped = img_array.reshape(-1, 1) / 255

    while True:
        color_1 = np.random.randint(256, size=3)
        color_2 = np.random.randint(256, size=3)

        # Enforce a minimum distance between them to make sure there's some visual contrast
        if np.linalg.norm(color_1 - color_2) > 100:
            break

    diff = (color_2 - color_1).reshape(1, -1)

    new_array = (color_1 + unwrapped @ diff).reshape(*img_array.shape, 3).astype(np.uint8)
    return new_array

def looks_grayscale(array, tolerance=2):

    if array.shape == 2:
        return True

    maxs = array.max(axis=2)
    mins = array.min(axis=2)
    return np.all(np.abs(maxs - mins) <= tolerance)

if __name__ == '__main__':
    input_folder = os.path.join(os.path.expanduser('~'), 'Documents', 'Textures')
    output_folder = os.path.join(os.path.expanduser('~'), 'Documents', 'TexturesProcessed')

    DESIRED_TEXTURE_SIZE = 256       # Assumes square, other images will be resized accordingly
    GREYSCALE_RECOLORS = 6          # For each greyscale image, generate new images

    all_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                all_files.append(os.path.join(root, file))

    counter = 0

    for i, file in enumerate(all_files):
        print('{}/{}'.format(i+1, len(all_files)))

        try:
            img = Image.open(file)
        except PIL.UnidentifiedImageError:
            print("Couldn't identify this file, skipping: {}".format(file))
            continue

        size = img.size

        if size[0] == size[1]:
            img = img.resize((DESIRED_TEXTURE_SIZE, DESIRED_TEXTURE_SIZE))
        else:
            scale = DESIRED_TEXTURE_SIZE / np.sqrt(size[0] * size[1])
            new_size = (int(size[0] * scale), int(size[1] * scale))
            img = img.resize(new_size)

        output_file = os.path.join(output_folder, f'{counter}.png')
        img.save(output_file)
        counter += 1

        img_array = np.array(img)
        for _ in range(GREYSCALE_RECOLORS):
            new_img_array = process_grayscale_image(img_array)
            output_file = os.path.join(output_folder, f'{counter}.png')
            Image.fromarray(new_img_array).save(output_file)
            counter += 1
