from pybullet_env import CutterEnv
import os
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

def blur(array, sigma):
    return gaussian_filter(array, sigma=(sigma, sigma, 0))

def convert_rgb_seg_to_output(rgb, seg):

    hsv = np.array(Image.fromarray(rgb).convert(mode='HSV'))
    seg_mask = 255 * seg
    hsv[:,:,1] = seg_mask
    return hsv



if __name__ == '__main__':

    width = 424
    height = 240
    grayscale = False
    use_seg = False
    use_depth = False
    use_flow = False
    use_last_frame = True
    output_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'TrainingImages')

    import sys
    args = sys.argv[1:]

    timesteps = 5000
    file_pref = ''

    if len(args) >= 1:
        timesteps = int(args[0])

    if len(args) >= 2:
        file_pref = '{}_'.format(args[1])

    env = CutterEnv(width, height, grayscale=grayscale, use_seg=use_seg, use_depth=use_depth, use_flow=use_flow,
                    use_last_frame=use_last_frame, use_gui=False, max_elapsed_time=1.0, max_vel=0.05, debug=True,
                    eval=False, difficulty=1.0)
    obs = env.reset()
    all_dists = []
    for i in range(timesteps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action, realtime=False)
        env.randomize_camera()


        random_img, _, _ = env.set_random()
        sigma = np.random.uniform(0.0, 1.5)

        raw_canonical_img, _, canonical_seg = env.set_canonical()
        canonical_img = convert_rgb_seg_to_output(raw_canonical_img, canonical_seg)
        random_img = blur(random_img, sigma)

        Image.fromarray(random_img).save(os.path.join(output_dir, 'randomized', f'{file_pref}{i}.png'))
        Image.fromarray(canonical_img).save(os.path.join(output_dir, 'canonical', f'{file_pref}{i}.png'))

        if done:
            all_dists.append(env.get_cutter_dist())
            obs = env.reset()

        print('{} / {}'.format(i+1, timesteps))
