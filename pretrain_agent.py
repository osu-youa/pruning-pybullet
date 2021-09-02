import numpy as np
import matplotlib.pyplot as plt
import os
from pybullet_env import CutterEnvBase, check_input_variance
from stable_baselines3 import PPO
# import cv2
import time
from PIL import Image
from pybullet_env import CutterEnv
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

data_root = 'training_data'

def model_eval(net, dataset, loss_func):
    net.eval()
    with torch.no_grad():
        running_loss = 0.0
        total_items = 0
        for inputs, truth in dataset:
            n = inputs.shape[0]
            total_items += n

            outputs = net(inputs.cuda())[0].cpu()
            running_loss += loss_func(outputs, truth).item() * n

    return running_loss / total_items



class ImgActionDataset(Dataset):

    def __init__(self, file_ids, truth_dict, unnormalize = True, preprocess_func=None):
        self.data = []
        for file_id in file_ids:
            truth_val = truth_dict[file_id]
            self.data.append([file_id, torch.from_numpy(truth_val).float()])
        self.unnormalize = unnormalize
        self.preprocess_func = preprocess_func

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        file_id, truth_val = self.data[idx]
        file_path = os.path.join(data_root, '{}.png'.format(file_id))
        img = Image.open(file_path)
        if self.preprocess_func is not None:
            img = np.array(img)
            img = Image.fromarray(self.preprocess_func(img))

        img_tensor = transforms.ToTensor()(img).unsqueeze_(0)[0]
        if self.unnormalize:
            img_tensor = img_tensor * 255

        return img_tensor, truth_val

if __name__ == '__main__':
    action = 'get_data'
    # action = 'train'
    # action = 'eval'
    difficulty = 1.0
    width = 424
    height = 240
    grayscale = False
    use_net = True
    use_seg = False
    use_depth = False
    use_flow = False
    use_last_frame = False
    crop = ((64,424), (60,240))
    downscale = (160, 80)

    actions_dict_file = os.path.join(data_root, 'actions.pickle')

    if action == 'get_data':

        env = CutterEnv(width, height, grayscale=grayscale, use_net=use_net, use_seg=use_seg, use_depth=use_depth,
                        use_flow=use_flow, use_last_frame=use_last_frame,
                        use_gui=True, max_elapsed_time=1.0, max_vel=0.3, debug=True,
                        eval=True, eval_count=None, crop=crop, downscale=downscale, difficulty=difficulty)
        model = PPO("CnnPolicy", env, verbose=1)

        current_img_actions = {}
        try:
            with open(actions_dict_file, 'rb') as fh:
                current_img_actions = pickle.load(fh)
        except FileNotFoundError:
            pass


        # model = model.load('best_model_0_5.zip')
        obs = env.reset()
        counter = 0
        steps_counter = 0
        while True:
            if not steps_counter % 1:
                while True:
                    target_action = input("What input is appropriate? (Type either UD, RL, nothing, or S to skip, X to reset): ").lower()
                    if ('u' in target_action and 'd' in target_action) or ('l' in target_action and 'r' in target_action):
                        print("Can't have opposing directions in input!")
                        continue
                    break
                if not ('s' in target_action or 'x' in target_action):
                    target_array = np.array([0.0, 0.0])
                    if 'u' in target_action:
                        target_array[1] = 1.0
                    if 'd' in target_action:
                        target_array[1] = -1.0
                    if 'l' in target_action:
                        target_array[0] = 1.0
                    if 'r' in target_action:
                        target_array[0] = -1.0

                    for _ in range(12):
                        env.randomize_camera()
                        env.randomize_tree_textures()
                        env.randomize_scenery()

                        rgb_array = env.get_obs()
                        while counter in current_img_actions:
                            counter += 1

                        Image.fromarray(rgb_array).save(os.path.join(data_root, '{}.png'.format(counter)))
                        current_img_actions[counter] = target_array

                    with open(actions_dict_file, 'wb') as fh:
                        pickle.dump(current_img_actions, fh)

                    # Array stats
                    summary = np.array(list(current_img_actions.values()))
                    means = summary.mean(axis=0)
                    print('Current means: {:.4f}, {:.4f}'.format(*means))
                else:
                    print('Skipped')

            steps_counter += 1



            # action, _states = model.predict(obs, deterministic=True)
            action = env.action_space.sample()
            env.set_action(action)
            obs, reward, done, info = env.step(action, realtime=False)
            if done or 'x' in target_action:
                obs = env.reset()


    elif action == 'train':

        env = CutterEnvBase(width, height, grayscale=grayscale, use_net=use_net, use_seg=use_seg, use_depth=use_depth,
                        use_flow=use_flow, use_last_frame=use_last_frame, use_gui=False, crop=crop, downscale=downscale)
        model = PPO("CnnPolicy", env, verbose=1)

        with open(actions_dict_file, 'rb') as fh:
            current_img_actions = pickle.load(fh)

        # data = ImgActionDataset([0], current_img_actions, preprocess_func=env.process_rgb_image)
        all_idxs = np.array([int(x.replace('.png', '')) for x in os.listdir(data_root) if x.endswith('.png')])
        training_truth_idx = np.ones(len(all_idxs), dtype=bool)

        # Pick 15% of the images to be part of the validation set
        n_val = int(len(all_idxs) * 0.15)
        training_truth_idx[np.random.choice(len(training_truth_idx), n_val, replace=False)] = False
        train_idx = all_idxs[training_truth_idx]
        val_idx = all_idxs[~training_truth_idx]
        print('VALIDATION IDX: {}'.format(val_idx))

        train_data = ImgActionDataset(train_idx, current_img_actions)
        val_data = ImgActionDataset(val_idx, current_img_actions)
        train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=0)

        net = model.policy.cuda()
        loss_func = nn.MSELoss()
        opt = optim.Adam(net.parameters(), lr=0.0005)

        init_loss = model_eval(net, val_loader, loss_func)
        print('Initial loss: {:.4f}'.format(init_loss))
        best_loss = init_loss

        for epoch in range(200):
            print('Starting epoch {}'.format(epoch+1))

            for i, data in enumerate(train_loader):
                net.train()
                inputs, truth = data
                opt.zero_grad()

                outputs = net(inputs.cuda())[0]
                loss = loss_func(outputs, truth.cuda())
                loss.backward()
                opt.step()

            eval_loss = model_eval(net, val_loader, loss_func)
            if eval_loss < best_loss:
                best_loss = eval_loss
                print('New best loss of {:.4f}'.format(eval_loss))
                torch.save(net.state_dict(), 'pretrained_agent.weights')
                # model.save('pretrained_agent.model')
            else:
                print('Loss was {:.4f}'.format(eval_loss))
            

    elif action == 'eval':

        env = CutterEnv(width, height, grayscale=grayscale, use_net=use_net, use_seg=use_seg, use_depth=use_depth, use_flow=use_flow, use_last_frame=use_last_frame,
                        use_gui=True, max_elapsed_time=2.0, max_vel=0.25, debug=True, eval=True, eval_count=None, difficulty=1.0, crop=crop, downscale=downscale)
        model = PPO("CnnPolicy", env, verbose=1)
        obs = env.reset()

        net = model.policy
        net.load_state_dict(torch.load('pretrained_agent.weights'))
        # net.eval()

        last_obs = None
        counter = 0
        while True:

            counter += 1
            if last_obs is not None:
                last_tree = last_obs[:,:,1].astype(int)
                curr_tree = obs[:,:,1].astype(int)

                tree_change = ((last_tree - curr_tree) / 2).astype(np.uint8)

            action = model.predict(obs, deterministic=True)[0]
            env.set_action(action)
            last_obs = obs
            obs, reward, done, info = env.step(action, realtime=False)
            # env.randomize_camera()
            if done:
                obs = env.reset()
