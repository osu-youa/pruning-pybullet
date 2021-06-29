import numpy as np
from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter

class PerlinNoiseBuffer:

    def __init__(self, img_width, img_height, rectangle_size=10, load_from_cache=True, buffer_size=10):
        self.buffer = [None] * buffer_size
        self.width = img_width
        self.height = img_height
        self.octave_size = (img_width / rectangle_size) + 0.0001    # Hack because of an unusual bug of some sort
        self.buffer_index = 0

        self.buffer_folder = os.path.join('noise_cache', '{}_{}_{}'.format(img_width, img_height, rectangle_size))
        self.load_from_cache = load_from_cache

        if load_from_cache:
            if not os.path.exists(self.buffer_folder):
                os.mkdir(self.buffer_folder)
            for i in range(buffer_size):
                noise_file = os.path.join(self.buffer_folder, '{}.npy'.format(i))
                if not os.path.exists(noise_file):
                    print('Generating {}...'.format(i))
                    self.generate_noise(output=noise_file, idx=i)
        else:
            print('Generating new noise...')
            for _ in range(buffer_size):
                self.generate_noise()

    def generate_noise(self, output=None, idx=None):

        if idx is not None:
            noise_generator = PerlinNoise(octaves=self.octave_size, seed=idx+1)
        else:
            noise_generator = PerlinNoise(octaves=self.octave_size)

        img = np.zeros(shape=(self.height, self.width))

        width_range = np.linspace(0, 1, num=self.width, endpoint=False)
        height_range = np.linspace(0, self.height / self.width, num=self.height, endpoint=False)
        for i, w in enumerate(width_range):
            for j, h in enumerate(height_range):
                img[j, i] = noise_generator([w, h])

        imax = np.max(img)
        imin = np.min(img)
        img = (img - imin) / (imax - imin)

        if output:
            np.save(output, img)
        else:
            self.buffer[self.buffer_index] = img
        self.buffer_index = (self.buffer_index + 1) % len(self.buffer)


    def retrieve_noise(self, idx=None):
        if idx is None:
            idx = self.buffer_index
        if self.load_from_cache:
            noise_file = os.path.join(self.buffer_folder, '{}.npy'.format(idx))
            return np.load(noise_file)
        else:
            return self.buffer[idx]

    def show_noise(self, idx=None):

        img = self.retrieve_noise(idx=idx)
        max_val = np.max(img)
        min_val = np.min(img)

        print('Max val: {:.4f}\nMin val: {:.4f}'.format(max_val, min_val))

        plt.imshow(img)
        plt.show()

    def get_random(self):
        return self.retrieve_noise(idx=np.random.randint(len(self.buffer)))

def overlay_noise(img, sigma, noise_mask, noise_weight=0.2, convert_to_uint8=False):

    if img.dtype == np.uint8:
        img = img / 255

    blurred_img = gaussian_filter(img, sigma)
    new_img = (1 - noise_weight) * img + noise_weight * blurred_img * noise_mask
    new_img[new_img > 1.0] = 1.0
    new_img[new_img < 0.0] = 0.0

    if convert_to_uint8:
        new_img = (new_img * 255).astype(np.uint8)

    return new_img


if __name__ == '__main__':
    buffer = PerlinNoiseBuffer(159, 90, 30, buffer_size=50)
    for i in range(10):
        buffer.show_noise(i)