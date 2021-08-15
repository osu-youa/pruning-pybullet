from pybullet_env import CutterEnv
import os
import PIL

if __name__ == '__main__':

    width = 424
    height = 240
    grayscale = False
    use_seg = False
    use_depth = False
    use_flow = False
    use_last_frame = True
    output_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'TrainingImages')

    timesteps = 10000

    env = CutterEnv(width, height, grayscale=grayscale, use_seg=use_seg, use_depth=use_depth, use_flow=use_flow,
                    use_last_frame=use_last_frame, use_gui=False, max_elapsed_time=1.0, max_vel=0.05, debug=True,
                    eval=False, difficulty=1.0)
    obs = env.reset()
    all_dists = []
    for i in range(timesteps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action, realtime=False)

        random_img = env.set_random()[0]
        canonical_img = env.set_canonical()[0]

        PIL.Image.fromarray(random_img).save(os.path.join(output_dir, 'randomized', f'{i}.png'))
        PIL.Image.fromarray(canonical_img).save(os.path.join(output_dir, 'canonical', f'{i}.png'))

        if done:
            all_dists.append(env.get_cutter_dist())
            obs = env.reset()
