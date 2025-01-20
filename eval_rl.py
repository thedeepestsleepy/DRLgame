import torch
import torch.nn as nn
import numpy as np
from environment import RedSquareEnv
import pygame
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision
from PIL import Image
import os

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

def get_object_detector(num_classes):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # print('in features', in_features.shape)
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

    positions.append(total_reward / 1000.0)  
    if len(positions) < 21:
        positions += [0.0] * (21 - len(positions))
    else:
        positions = positions[:21]
    return torch.tensor(positions, dtype=torch.float32).unsqueeze(0).to(device)

def evaluate(num_episodes=1000):
    env = RedSquareEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_actions = env.action_space.n
    input_dim = 21  
    policy_net = DuelingDQN(input_dim, num_actions).to(device)
    policy_net.load_state_dict(torch.load("custom_rl_model.pth", map_location=device))
    policy_net.eval()


    detection_model = get_object_detector(num_classes=4).to(device)
    detection_model.eval()

    rewards = []

    save_dir = 'eval_rl_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        state_tensor = preprocess_state(state, detection_model, device, total_reward)
        time_step = 0
        done = False

        while not done:
            time_step +=1
    
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = q_values.max(1)[1].item()

   
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            next_state_tensor = preprocess_state(next_state, detection_model, device,total_reward)

            state_tensor = next_state_tensor
            if time_step == 1000:
                done = True
            # if done:
            #     
            #     if total_reward < 200:
            #         
            #         frame = next_state  
            #      
            #         image = Image.fromarray(frame)
            #         
            #         image_path = os.path.join(save_dir, f'episode_{episode + 1}_reward_{int(total_reward)}.png')
            #         image.save(image_path)
            #         print(f"Saved state image for episode {episode + 1} with total reward {total_reward}")
          
            for event in pygame.event.get():
                
                if event.type == pygame.QUIT:
                    
                    env.close()
                    pygame.quit()
                    exit()

            

        rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

        

    env.close()
    pygame.quit()


    np.save("evaluation_rewards.npy", np.array(rewards))


    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(rewards)), rewards, alpha=0.7, s=3)
    plt.title("Evaluation Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig("Evaluation_rewards.png")
    plt.show()

if __name__ == "__main__":
    evaluate()
