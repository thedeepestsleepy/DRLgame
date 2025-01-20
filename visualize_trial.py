import torch
import torch.nn as nn
import numpy as np
from environment import RedSquareEnv
import pygame
import time
import torchvision.transforms as T
import torchvision


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


def visualize():
    env = RedSquareEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_actions = env.action_space.n
    input_dim = 21  
    policy_net = DuelingDQN(input_dim, num_actions).to(device)
    policy_net.load_state_dict(torch.load("custom_rl_model.pth", map_location=device))
    policy_net.eval()


    detection_model = get_object_detector(num_classes=4).to(device)
    detection_model.eval()


    pygame.font.init()
    font = pygame.font.SysFont('Arial', 30)

    state = env.reset()
    total_reward = 0
    state_tensor = preprocess_state(state, detection_model, device, total_reward)
    done = False
    
    start_time = time.time()
    game_over = False

    while True:
        if not game_over:
 
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = q_values.max(1)[1].item()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            next_state_tensor = preprocess_state(next_state, detection_model, device,total_reward)

            state_tensor = next_state_tensor


            env.render(mode='background')


            if done:
                end_time = time.time()
                total_time = end_time - start_time
                game_over = True
        else:
          
            env.render(mode='background')
        
            message = f"Time cost: {total_time:.2f} Seconds, Total Rewards: {total_reward:.2f}"
            text_surface = font.render(message, True, (255, 255, 255))

            env.display_screen.blit(
                text_surface,
                ((env.WIDTH - text_surface.get_width()) // 2, (env.HEIGHT - text_surface.get_height()) // 2)
            )

            pygame.display.flip()


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                exit()


        env.clock.tick(60)

if __name__ == "__main__":
    visualize()
