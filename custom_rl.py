import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision
import pygame
from collections import deque
from environment import RedSquareEnv
import matplotlib.pyplot as plt

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )
     
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
      
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
      
        qvals = value + (advantage - advantage.mean())
        return qvals

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def save(self, path):

        torch.save(self.memory, path)

    def load(self, path):

        self.memory = torch.load(path)

def get_object_detector(num_classes):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load('fasterrcnn_resnet50_fpn.pth', map_location='cpu'))
    model.eval()
    return model

def preprocess_state(state, model, device, total_reward):

    transform = T.Compose([T.ToTensor()])
    img = transform(state).to(device)
    with torch.no_grad():
        outputs = model([img])
    positions = []
    for output in outputs:
        boxes = output['boxes']
        labels = output['labels']
     
        for i in range(len(labels)):
            label = labels[i].item()
            box = boxes[i].cpu().numpy()
    
            x_center = (box[0] + box[2]) / 2 / state.shape[1]
            y_center = (box[1] + box[3]) / 2 / state.shape[0]
            width = (box[2] - box[0]) / state.shape[1]
            height = (box[3] - box[1]) / state.shape[0]
            positions.extend([label / 4, x_center, y_center, width, height])

    positions.append(total_reward / 100.0)  
    if len(positions) < 21:
        positions += [0.0] * (21 - len(positions))
    else:
        positions = positions[:21]
    return torch.tensor(positions, dtype=torch.float32).unsqueeze(0).to(device)

def train():
  
    env = RedSquareEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_actions = env.action_space.n
    input_dim = 21  
    policy_net = DuelingDQN(input_dim, num_actions).to(device)
    target_net = DuelingDQN(input_dim, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = ReplayMemory(50000)

    num_episodes = 1000  
    batch_size = 64
    gamma = 0.99
    target_update = 5

    epsilon_start = 1.0
    epsilon_end = 0.01  
    # epsilon_decay = 2000

    phase1_end = int(num_episodes * 0.1)
    phase2_end = int(num_episodes * 0.6)

    rewards = []

    detection_model = get_object_detector(num_classes=4).to(device)
    detection_model.eval()

    checkpoint_path = "rl_checkpoint.pth"
    start_episode = 0
    steps_done = 0
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        target_net.load_state_dict(checkpoint['target_net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        memory.memory = checkpoint['memory']
        rewards = checkpoint['rewards']
        start_episode = checkpoint['episode'] + 1
        steps_done = checkpoint['steps_done']
        print(f"Resuming training from episode {start_episode}")

    for episode in range(start_episode, num_episodes):
        state = env.reset()
        total_reward = 0  
        state_tensor = preprocess_state(state, detection_model, device, total_reward)
        done = False
        time_steps = 0

        while not done:
      
            if episode < phase1_end:
                # Phase 1: low decay
                epsilon = max(epsilon_end, epsilon_start - (epsilon_start - 0.6) * (episode / phase1_end))
            elif episode < phase2_end:
                # Phase 2: Moderate decay
                epsilon = max(epsilon_end, 0.6 - (0.6 - 0.2) * ((episode - phase1_end) / (phase2_end - phase1_end)))
            else:
                # Phase 3: Fast decay
                epsilon = max(epsilon_end, 0.2 - (0.2 - epsilon_end) * ((episode - phase2_end) / (num_episodes - phase2_end)))

            steps_done += 1
            time_steps += 1

            if random.random() < epsilon:
                action = random.randrange(num_actions)
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action = q_values.max(1)[1].item()


            next_state, reward, done, info = env.step(action)
            total_reward += reward
            next_state_tensor = preprocess_state(next_state, detection_model, device, total_reward)
            if time_steps > 1000:
                done = True

            reward_centering = reward - total_reward/time_steps # reward centering
            memory.push((state_tensor, action, reward_centering, next_state_tensor, done))

            state_tensor = next_state_tensor

   
            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

                batch_state = torch.cat(batch_state).to(device)
                batch_action = torch.tensor(batch_action).unsqueeze(1).to(device)
                batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(1).to(device)
                batch_next_state = torch.cat(batch_next_state).to(device)
                batch_done = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1).to(device)
                current_q_values = policy_net(batch_state).gather(1, batch_action)
                # print('current q values',current_q_values)

                with torch.no_grad():
                    next_actions = policy_net(batch_next_state).max(1)[1].unsqueeze(1)
                    next_q_values = target_net(batch_next_state).gather(1, next_actions)
                    expected_q_values = batch_reward + (gamma * next_q_values * (1 - batch_done))
                    # print('expected_q_values',expected_q_values)

                loss = nn.functional.mse_loss(current_q_values, expected_q_values)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    sys.exit()

        rewards.append(total_reward)

        if (episode + 1) % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")


        if (episode + 1) % 50 == 0:
            checkpoint = {
                'episode': episode,
                'policy_net_state_dict': policy_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'memory': memory.memory,
                'rewards': rewards,
                'steps_done': steps_done
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at episode {episode + 1}")
            torch.save(policy_net.state_dict(), "custom_rl_model.pth")



    np.save("training_rewards.npy", np.array(rewards))


    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(rewards)), rewards, alpha=0.7, s=3)
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig("training_rewards.png")
    plt.show()



    env.close()

if __name__ == "__main__":
    train()
