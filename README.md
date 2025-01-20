**Installation**

pip install torch torchvision pymunk pygame numpy opencv-python matplotlib
or
pip install -r requirements.txt



**Ensure all scripts are run in the specified order**

Dataset Generation:

    python generate_data.py
    
Training the Object Detection Model:

    python train_detector.py
    
Training the Reinforcement Learning Agent:

    python custom_rl.py
    
Visualizing the Agent:

    python visualize_trial.py
    
Play by yourself:

    python visualize.py

or you can run all:

chmod +x run_all.sh

./run_all.sh






Project Files

    generate_data.py: Generates synthetic datasets.
    
    train_detector.py: Trains the object detection model.
    
    custom_rl.py: Trains the RL agent using Dueling DQN.
    
    eval_rl.py: Evaluates the RL agent’s performance.
    
    eval_detector.py: Evaluates the object detection model.
    
    visualize.py: Allows manual control of the environment.
    
    visualize_trial.py: Displays the RL agent in action.

Results and Outputs

    Object Detection Model: Trained weights saved as fasterrcnn_resnet50_fpn.pth.
    
    RL Agent Model: Trained weights saved as custom_rl_model.pth.


