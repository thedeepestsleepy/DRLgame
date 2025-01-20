#!/bin/bash

# Step 1: Generate dataset
echo "Generating dataset..."
python generate_data.py

# Step 2: Train object detection model
echo "Training object detection model..."
python train_detector.py

# Step 3: Evaluate object detection model
echo "Evaluating object detection model..."
python eval_detector.py

# Step 4: Train reinforcement learning agent
echo "Training reinforcement learning agent..."
python custom_rl.py

# Step 5: Evaluate RL agent
echo "Evaluating RL agent..."
python eval_rl.py


# Step 6: Visualize RL agent
echo "Visualizing RL agent..."
python visualize_trial.py
