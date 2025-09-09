import logging
logging.getLogger().setLevel(logging.CRITICAL)
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import pickle
import time
from symphony_op_2_1 import Symphony, ReplayBuffer
import math
import sys
import re
import os

"""

python train.py --test-only --episodes=100

python train.py --test-only --render --record_video --models-dir=final_test_models --episodes=100



"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# -------------------- Command line options and global parameters --------------------
option = 8
burst = False # big amplitude random moves in the beginning
tr_noise = True  #if extra noise is needed during training
explore_time = 5000
tr_between_ep_init = 15 # training between episodes
tr_between_ep_const = False
tr_per_step = 3 # training per frame/step
start_test = 250
limit_step = 2000 #max steps per episode
limit_eval = 2000 #max steps per evaluation
num_episodes = 10000000
start_episode = 0 #number for the identification of the current episode
total_rewards, total_steps, test_rewards, Q_learning = [], [], [], False
training_validation_episodes = 5 #number of episodes during validation
hidden_dim = 256
max_action = 1.0
fade_factor = 7 # fading memory factor, 7 -remembers ~30% of the last transtions before gradual forgetting, 1 - linear forgetting, 10 - ~50% of transitions, 100 - ~70% of transitions.
stall_penalty = 0.07 # moving is life, stalling is dangerous, optimal value = 0.07, higher values can create extra vibrations.
capacity = "full" # short = 100k, medium=300k, full=500k replay buffer memory size.

render = '--render' in sys.argv
record_video = '--record_video' in sys.argv
models_dir = os.getcwd()
test_episodes = 10
for arg in sys.argv:
    if arg.startswith('--models-dir='):
        models_dir = arg.split('=', 1)[1]
    match = re.match(r'--episodes=(\d+)', arg)
    if match:
        test_episodes = int(match.group(1))
        break
# Check for existence of models and buffer in current directory, else use 'final_test_models'
def models_and_buffer_exist(directory):
    required_files = ['actor_model.pt', 'critic_model.pt', 'critic_target_model.pt', 'replay_buffer']
    return all(os.path.exists(os.path.join(directory, f)) for f in required_files)

if not models_and_buffer_exist(models_dir):
    alt_dir = os.path.join(os.getcwd(), 'final_test_models')
    if models_and_buffer_exist(alt_dir):
        print(f"Models and buffer not found in {models_dir}, switching to {alt_dir}")
        models_dir = alt_dir
    else:
        print(f"Warning: Models and buffer not found in {models_dir} or {alt_dir}")
print(f"Using models directory: {models_dir}")

# -------------------- Environment setup --------------------
if option == -1:
    env = gym.make('Pendulum-v1')
    env_test = gym.make('Pendulum-v1', render_mode="human")
elif option == 0:
    env = gym.make('MountainCarContinuous-v0')
    env_test = gym.make('MountainCarContinuous-v0', render_mode="human")
elif option == 1:
    env = gym.make('HalfCheetah-v4')
    env_test = gym.make('HalfCheetah-v4', render_mode="human")
elif option == 2:
    tr_between_ep_init = 70
    env = gym.make('Walker2d-v4')
    env_test = gym.make('Walker2d-v4', render_mode="human")
elif option == 3:
    tr_between_ep_init = 200
    env = gym.make('Humanoid-v4')
    env_test = gym.make('Humanoid-v4', render_mode="human")
elif option == 4:
    limit_step = 300
    limit_eval = 300
    tr_between_ep_init = 70
    env = gym.make('HumanoidStandup-v4')
    env_test = gym.make('HumanoidStandup-v4', render_mode="human")
elif option == 5:
    env = gym.make('Ant-v4')
    env_test = gym.make('Ant-v4', render_mode="human")
    angle_limit = 0.4
    max_action = 0.7
elif option == 6:
    tr_between_ep_init = 40
    burst = True
    tr_noise = False
    limit_step = int(1e+6)
    env = gym.make('BipedalWalker-v3')
    env_test = gym.make('BipedalWalker-v3', render_mode="human")
elif option == 7:
    burst = True
    tr_noise = False
    tr_between_ep_init = 0
    env = gym.make('BipedalWalkerHardcore-v3', render_mode="human")
    env_test = gym.make('BipedalWalkerHardcore-v3')
elif option == 8:
    limit_step = 500
    limit_eval = 500
    env = gym.make('LunarLanderContinuous-v3')
    env_test = gym.make('LunarLanderContinuous-v3', render_mode="human")
elif option == 9:
    limit_step = 300
    limit_eval = 200
    env = gym.make('Pusher-v4')
    env_test = gym.make('Pusher-v4', render_mode="human")
elif option == 10:
    burst = True
    env = gym.make('Swimmer-v4')
    env_test = gym.make('Swimmer-v4', render_mode="human")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print('action space high', env.action_space.high)
max_action = max_action*torch.FloatTensor(env.action_space.high).to(device) if env.action_space.is_bounded() else max_action*1.0
replay_buffer = ReplayBuffer(state_dim, action_dim, capacity, device, fade_factor, stall_penalty)
algo = Symphony(state_dim, action_dim, hidden_dim, device, max_action, burst, tr_noise)




#used to create random initalization in Actor -> less dependance on the specific random seed.
def init_weights(m):
    if isinstance(m, nn.Linear): torch.nn.init.xavier_uniform_(m.weight)




#testing model
def testing(env, limit_step, test_episodes, render=False, record_video=False, video_dir="videos"):
    """
    Test the agent in the environment.
    If record_video is True, saves a video for each test_episode in video_dir.
    """
    if test_episodes < 1:
        return
    print("Validation... ", test_episodes, " episodes")
    episode_return = []

    import random
    import os
    last_rewards = []

    for test_episode in range(test_episodes):
        # Use test_episode as the seed for reproducibility per episode
        seed = test_episode + 1
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        rewards = []
        last_reward = None

        # Optionally wrap env for video recording (after knowing the episode return)
        # But we need the return value for the filename, so we must delay wrapping until after the episode
        if hasattr(env, 'reset'):
            state = env.reset(seed=seed)[0]
        else:
            state = env.reset()[0]

        for steps in range(1, limit_step + 1):
            action = algo.select_action(state, replay_buffer, mean=True)
            next_state, reward, done, info, _ = env.step(action)
            rewards.append(reward)
            last_reward = reward
            state = next_state
            if done:
                break

        ep_return = np.sum(rewards)
        episode_return.append(ep_return)
        last_rewards.append(last_reward)

        validate_return = np.mean(episode_return[-100:])
        print(f"trial {test_episode}:, Rtrn = {ep_return:.2f}, Average 100 = {validate_return:.2f}, steps: {steps}, Last reward: {last_reward}")

        # Now record the video if requested, replaying the episode with the same seed and actions
        if record_video:
            os.makedirs(video_dir, exist_ok=True)
            from gymnasium.wrappers import RecordVideo
            # Re-create the environment for video recording
            video_name = f"test_ep{test_episode}_return_{ep_return:.2f}"
            video_env = RecordVideo(make_env_test(option, render, record_video=True), video_dir, episode_trigger=lambda ep: True, name_prefix=video_name)
            # Replay the episode
            if hasattr(video_env, 'reset'):
                state = video_env.reset(seed=seed)[0]
            else:
                state = video_env.reset()[0]
            for a in rewards:
                # This is a hack: we don't have the actions, only rewards, so just run the episode again
                # In practice, to get the exact same episode, you would need to store actions and states
                # Here, we just run the episode again with the same seed
                action = algo.select_action(state, replay_buffer, mean=True)
                next_state, reward, done, info, _ = video_env.step(action)
                state = next_state
                if done:
                    break
            try:
                video_env.close()
            except Exception as e:
                print(f"Warning: Could not close video_env for video saving: {e}")

    # After all episodes, log the statistics of last_rewards
    count_100 = sum(1 for r in last_rewards if r == 100)
    count_neg100 = sum(1 for r in last_rewards if r == -100)
    count_near0 = sum(1 for r in last_rewards if abs(r) < 10)
    print(f"Summary of last rewards over {test_episodes} episodes: +100={count_100}, -100={count_neg100}, ~0={count_near0}")

    # Collect model details
    def count_nodes(module):
        nodes = 0
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nodes += m.out_features
        return nodes

    def get_model_details(model, name):
        total_params = sum(p.numel() for p in model.parameters())
        nodes = count_nodes(model)
        return {"name": name, "total_parameters": total_params, "total_nodes": nodes}

    actor_details = get_model_details(algo.actor, "Actor")
    critic_details = get_model_details(algo.critic, "Critic")

    total_parameters = actor_details["total_parameters"] + critic_details["total_parameters"]
    total_nodes = actor_details["total_nodes"] + critic_details["total_nodes"]
    results_json = {
        "model_details": [actor_details, critic_details],
        "total_parameters": total_parameters,
        "total_nodes": total_nodes,
        "num_episodes": test_episodes,
        "count_100": count_100,
        "count_neg100": count_neg100,
        "count_near0": count_near0
    }
    import json
    print("TEST_RESULTS_JSON_START\n" + json.dumps(results_json) )

    rtn = test_episodes == count_100

    if rtn:
        print("All test episodes achieved a reward of 100.")

    if record_video:
        # Ensure video is saved by closing the environment
        try:
            env.close()
        except Exception as e:
            print(f"Warning: Could not close env for video saving: {e}")
    return rtn

#--------------------loading existing models, replay_buffer, parameters-------------------------

try:
    print("loading buffer...")
    try:
        buffer_path = os.path.join(models_dir, 'replay_buffer')
        with open(buffer_path, 'rb') as file:
            dict = pickle.load(file)
    except Exception as e:
        import sys, os
        if '--test-only' in sys.argv:
            print(f"Trying alternate buffer path: {buffer_path}")
            raise
        else:
            raise e
    algo.actor.noise.x_coor = dict['x_coor']
    replay_buffer = dict['buffer']
    total_rewards = dict['total_rewards']
    total_steps = dict['total_steps']
    average_steps = dict['average_steps']
    if len(replay_buffer)>=explore_time and not Q_learning: Q_learning = True
    print('buffer loaded, buffer length', len(replay_buffer))

    start_episode = len(total_steps)

except Exception as e:
    print("problem during loading buffer")

#-------------------------------------------------------------------------------------

# Dynamically create env_test for test-only mode, with or without rendering
import sys
import re
render = '--render' in sys.argv
# Add record_video command line option
record_video = '--record_video' in sys.argv
# Add models directory command line option



# Always print the models_dir being used (after possible switch)

def make_env_test(option, render, record_video=False):
    # If recording video, use render_mode="rgb_array" as required by gymnasium RecordVideo
    if record_video:
        render_mode = "rgb_array"
    else:
        render_mode = "human" if render else None
    if option == -1:
        return gym.make('Pendulum-v1', render_mode=render_mode)
    elif option == 0:
        return gym.make('MountainCarContinuous-v0', render_mode=render_mode)
    elif option == 1:
        return gym.make('HalfCheetah-v4', render_mode=render_mode)
    elif option == 2:
        return gym.make('Walker2d-v4', render_mode=render_mode)
    elif option == 3:
        return gym.make('Humanoid-v4', render_mode=render_mode)
    elif option == 4:
        return gym.make('HumanoidStandup-v4', render_mode=render_mode)
    elif option == 5:
        return gym.make('Ant-v4', render_mode=render_mode)
    elif option == 6:
        return gym.make('BipedalWalker-v3', render_mode=render_mode)
    elif option == 7:
        return gym.make('BipedalWalkerHardcore-v3', render_mode=render_mode)
    elif option == 8:
        return gym.make('LunarLanderContinuous-v3', render_mode=render_mode)
    elif option == 9:
        return gym.make('Pusher-v4', render_mode=render_mode)
    elif option == 10:
        return gym.make('Swimmer-v4', render_mode=render_mode)
    else:
        raise ValueError("Unknown option for environment")

if '--test-only' in sys.argv:

    env_test = make_env_test(option, render, record_video=record_video)
    print('Running in test-only mode...')
    print(f"Render mode: {render}")
    print(f"Test episodes: {test_episodes}")
    print(f"Record video: {record_video}")

    # --- Model loading (using models_dir) ---
    try:
        print("loading models...")
        actor_path = os.path.join(models_dir, 'actor_model.pt')
        critic_path = os.path.join(models_dir, 'critic_model.pt')
        critic_target_path = os.path.join(models_dir, 'critic_target_model.pt')
        algo.actor.load_state_dict(torch.load(actor_path))
        algo.critic.load_state_dict(torch.load(critic_path))
        algo.critic_target.load_state_dict(torch.load(critic_target_path))
        print('models loaded')
    except Exception as e:
        print("problem during loading models")

    # --- Model details report ---
    def count_nodes(module):
        nodes = 0
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nodes += m.out_features
        return nodes

    def print_model_details(model, name):
        total_params = sum(p.numel() for p in model.parameters())
        nodes = count_nodes(model)
        print(f"{name} - Total parameters: {total_params}, Total nodes: {nodes}")

    print_model_details(algo.actor, "Actor")
    print_model_details(algo.critic, "Critic")

    # Try to generate network visualizations
    dummy_state = torch.zeros(1, state_dim).to(device)
    dummy_action = torch.zeros(1, action_dim).to(device)
    
    # Method 1: torchview (layer-by-layer architecture view)
    try:
        from torchview import draw_graph
        
        # Actor network visualization
        actor_model_graph = draw_graph(
            algo.actor, 
            input_data=dummy_state,
            expand_nested=True,
            save_graph=True,
            filename='actor_architecture',
            directory='.',
        )
        print("Actor architecture saved as actor_architecture.png")
        
        # Critic network visualization
        critic_model_graph = draw_graph(
            algo.critic,
            input_data=[dummy_state, dummy_action],
            expand_nested=True, 
            save_graph=True,
            filename='critic_architecture',
            directory='.',
        )
        print("Critic architecture saved as critic_architecture.png")
        
    except ImportError:
        print("torchview not installed. Run 'pip install torchview' to generate architecture diagrams.")
    except Exception as e:
        print(f"Could not generate architecture diagrams with torchview: {e}")
    
    # Method 2: torchviz (computational graph view)
    try:
        from torchviz import make_dot
        actor_out = algo.actor(dummy_state, mean=True)
        if isinstance(actor_out, (list, tuple)):
            actor_out = actor_out[0]
        actor_graph = make_dot(actor_out, params={k: v for k, v in algo.actor.named_parameters()})
        actor_graph.format = "png"
        actor_graph.render("actor_computation_graph", cleanup=True)
        print("Actor computation graph saved as actor_computation_graph.png")

        critic_out = algo.critic(dummy_state, dummy_action)
        if isinstance(critic_out, (list, tuple)):
            critic_out = critic_out[0]
        critic_graph = make_dot(critic_out, params={k: v for k, v in algo.critic.named_parameters()})
        critic_graph.format = "png"
        critic_graph.render("critic_computation_graph", cleanup=True)
        print("Critic computation graph saved as critic_computation_graph.png")
    except ImportError:
        print("torchviz not installed. Run 'pip install torchviz' to generate computation graphs.")
    except Exception as e:
        print(f"Could not generate computation graphs: {e}")
    
    # Method 3: Traditional neural network diagrams (nodes as circles)
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import networkx as nx
        
        def draw_traditional_nn(model, input_size, filename, title, width_scale=5, height_scale=1, max_nodes_per_layer=20, circle_size=0.15, line_thickness=0.5):
            """Draw traditional neural network with circles for nodes and lines for connections"""
            
            # Get layer sizes by analyzing the model
            layer_sizes = []
            layer_sizes.append(input_size)  # Input layer
            
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    layer_sizes.append(module.out_features)
            
            # Apply node scaling to keep visualization manageable
            original_layer_sizes = layer_sizes.copy()
            scaled_layer_sizes = []
            for size in layer_sizes:
                if size > max_nodes_per_layer:
                    scaled_size = max_nodes_per_layer
                    scaled_layer_sizes.append(scaled_size)
                else:
                    scaled_layer_sizes.append(size)
            
            # Calculate figure dimensions based on scaled network size
            num_layers = len(scaled_layer_sizes)
            max_nodes = max(scaled_layer_sizes)
            
            # Adjust figure size - make it wider and proportional to network size
            fig_width = max(8, num_layers * 3) * width_scale  # Apply width scale
            fig_height = max(6, max_nodes * 0.3) * height_scale  # Apply height scale
            print(f"Creating {title}: fig_width={fig_width}, fig_height={fig_height}, num_layers={num_layers}, max_nodes={max_nodes}, width_scale={width_scale}, height_scale={height_scale}")
            print(f"circle_size={circle_size}, line_thickness={line_thickness}")
            print(f"Original layer sizes: {original_layer_sizes}")
            print(f"Scaled layer sizes: {scaled_layer_sizes} (max_nodes_per_layer={max_nodes_per_layer})")
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
            
            # Calculate stretched x-coordinates to utilize the wider figure
            x_stretch_factor = width_scale  # Use the width scale as stretch factor
            x_spacing = 1 * x_stretch_factor  # Space between layers
            
            # Set limits to accommodate stretched x-coordinates
            ax.set_xlim(-0.5 * x_stretch_factor, (num_layers - 1) * x_stretch_factor + 0.5 * x_stretch_factor)
            ax.set_ylim(-1, max_nodes + 1)
            ax.set_aspect('equal')  # Keep circles perfectly round
            ax.axis('off')
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            # Colors for different layer types
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
            
            node_positions = {}
            
            # Draw nodes (circles) for each layer using scaled sizes
            for layer_idx, (original_size, scaled_size) in enumerate(zip(original_layer_sizes, scaled_layer_sizes)):
                color = colors[layer_idx % len(colors)]
                x = layer_idx * x_stretch_factor  # Apply horizontal stretching
                
                # Calculate y positions to center the layer
                if scaled_size == 1:
                    y_positions = [max_nodes / 2]
                else:
                    y_step = (max_nodes - 1) / max(1, scaled_size - 1) if scaled_size > 1 else 0
                    y_start = (max_nodes - 1 - y_step * (scaled_size - 1)) / 2
                    y_positions = [y_start + i * y_step for i in range(scaled_size)]
                
                # Keep circles at a reasonable size using the parameter
                circle_radius = circle_size
                
                # Draw nodes
                for node_idx, y in enumerate(y_positions):
                    circle = patches.Circle((x, y), circle_radius, facecolor=color, edgecolor='black', linewidth=1)
                    ax.add_patch(circle)
                    node_positions[(layer_idx, node_idx)] = (x, y)
                    
                    # Add node labels for small networks
                    if scaled_size <= 10:
                        ax.text(x, y, str(node_idx), ha='center', va='center', fontsize=8, fontweight='bold')
                    
                # Add "..." indicator if layer was truncated
                if original_size > scaled_size:
                    # Add ellipsis to indicate more nodes exist
                    ellipsis_y = y_positions[-1] + (y_positions[1] - y_positions[0]) if len(y_positions) > 1 else y_positions[0] + 0.5
                    ax.text(x, ellipsis_y, '...', ha='center', va='center', fontsize=12, fontweight='bold', style='italic')
            
            # Draw connections between layers (using scaled sizes)
            for layer_idx in range(len(scaled_layer_sizes) - 1):
                current_layer_size = scaled_layer_sizes[layer_idx]
                next_layer_size = scaled_layer_sizes[layer_idx + 1]
                
                # Draw lines between all nodes in adjacent layers
                for i in range(current_layer_size):
                    for j in range(next_layer_size):
                        if (layer_idx, i) in node_positions and (layer_idx + 1, j) in node_positions:
                            x1, y1 = node_positions[(layer_idx, i)]
                            x2, y2 = node_positions[(layer_idx + 1, j)]
                            
                            # Make lines semi-transparent to avoid clutter, use parameter for thickness
                            ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=line_thickness)
            
            # Add layer labels with more space, showing original sizes
            layer_names = ['Input'] + [f'Hidden {i}' for i in range(len(scaled_layer_sizes) - 2)] + ['Output']
            for i, (original_size, scaled_size, name) in enumerate(zip(original_layer_sizes, scaled_layer_sizes, layer_names)):
                if original_size == scaled_size:
                    label_text = f'{name}\n({original_size} nodes)'
                else:
                    label_text = f'{name}\n({original_size} nodes)\n(showing {scaled_size})'
                x_label = i * x_stretch_factor  # Use stretched x-coordinates for labels
                ax.text(x_label, -0.8, label_text, ha='center', va='top', 
                       fontsize=10, fontweight='bold')
            
            # Use tight_layout with padding to prevent squashing
            plt.tight_layout(pad=2.0)
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Traditional neural network diagram saved as {filename}")
        
        # Draw Actor network
        width_scale=10 
        height_scale=1
        max_nodes_per_layer=25
        circle_size=0.5
        line_thickness=1.0
        draw_traditional_nn(algo.actor, state_dim, 'actor_traditional_nn.png', 'Actor Network Architecture', width_scale=width_scale, height_scale=height_scale, max_nodes_per_layer=max_nodes_per_layer, circle_size=circle_size, line_thickness=line_thickness)

        # Draw Critic network  
        draw_traditional_nn(algo.critic, state_dim + action_dim, 'critic_traditional_nn.png', 'Critic Network Architecture', width_scale=width_scale, height_scale=height_scale, max_nodes_per_layer=max_nodes_per_layer, circle_size=circle_size, line_thickness=line_thickness)

    except ImportError as e:
        print(f"matplotlib not installed. Run 'pip install matplotlib networkx' for traditional NN diagrams: {e}")
    except Exception as e:
        print(f"Could not generate traditional NN diagrams: {e}")
    
    # Method 4: Simple text summary
    try:
        from torchsummary import summary
        print("\n=== ACTOR MODEL SUMMARY ===")
        summary(algo.actor, input_size=(state_dim,))
        print("\n=== CRITIC MODEL SUMMARY ===") 
        # For critic, we need to handle dual inputs
        print("Critic input: state + action")
        print(f"State shape: ({state_dim},), Action shape: ({action_dim},)")
    except ImportError:
        print("torchsummary not installed. Run 'pip install torchsummary' for detailed model summaries.")
    except Exception as e:
        print(f"Could not generate model summaries: {e}")

    testing(env_test, limit_eval, test_episodes, render=render, record_video=record_video)
    print(f"Test-only run complete. Models directory used: {models_dir}")
    exit()

#-------------------------------------------------------------------------------------

for i in range(start_episode, num_episodes):
    
    episode_start_time = time.time()
    rewards = []
    state = env.reset()[0]

    #----------------------------pre-processing------------------------------
    #---------------------0. increase ep training: -------------------------
    rb_len = len(replay_buffer)
    rb_len_treshold = 5000*tr_between_ep_init
    tr_between_ep = tr_between_ep_init
    if not tr_between_ep_const and tr_between_ep_init>=100 and rb_len>=350000: tr_between_ep = rb_len//5000 # init -> 70 -> 100
    if not tr_between_ep_const and tr_between_ep_init<100 and rb_len>=rb_len_treshold: tr_between_ep = rb_len//5000# init -> 100
    #---------------------------1. processor releave --------------------------
    if Q_learning: time.sleep(0.5)
    #---------------------2. decreases dependence on random seed: ---------------
    if not Q_learning and rb_len<explore_time:
        algo.actor.apply(init_weights)
    #-----------3. slighlty random initial configuration as in OpenAI Pendulum----
    action = 0.3*max_action.to('cpu').numpy()*np.random.uniform(-1.0, 1.0, size=action_dim)
    for steps in range(0, 2):
        next_state, reward, done, info, _ = env.step(action)
        rewards.append(reward)
        state = next_state
    
    

    #------------------------------training------------------------------

    if Q_learning: _ = [algo.train(replay_buffer.sample()) for x in range(tr_between_ep)]
        
    episode_steps = 0
    for steps in range(1, limit_step+1):
        episode_steps += 1

        if len(replay_buffer)>=explore_time and not Q_learning:
            replay_buffer.find_min_max()
            print("started training")
            Q_learning = True
            _ = [algo.train(replay_buffer.sample(uniform=True)) for x in range(64)]
            _ = [algo.train(replay_buffer.sample()) for x in range(64)]

        action = algo.select_action(state, replay_buffer)
        #action = algo.select_action(state)
        next_state, reward, done, info, _ = env.step(action)
        rewards.append(reward)
        
        #==============counters issues with environments================
        #(scores in report are not affected)
        #Ant environment has problem when Ant is flipped upside down and it is not detected (rotation around x is not checked), we can check to save some time:
        if env.spec.id.find("Ant") != -1:
            if (next_state[1]<angle_limit): done = True
            if next_state[1]>1e-3: reward += math.log(next_state[1]) #punish for getting unstable.
        #Humanoid-v4 environment does not care about torso being in upright position, this is to make torso little bit upright (z↑)
        elif env.spec.id.find("Humanoid-") != -1:
            reward += next_state[0]
        #fear less of falling/terminating. This Environments has a problem when agent stalls due to the high risks prediction. We decrease risks to speed up training.
        elif env.spec.id.find("LunarLander") != -1:
            if reward==-100.0: reward = -50.0
        #fear less of falling/terminating. This Environments has a problem when agent stalls due to the high risks prediction. We decrease risks to speed up training.
        elif env.spec.id.find("BipedalWalkerHardcore") != -1:
            if reward==-100.0: reward = -25.0
        #===============================================================
        
        replay_buffer.add(state, action, reward+1.0, next_state, done)
        if Q_learning: _ = [algo.train(replay_buffer.sample()) for x in range(tr_per_step)]
        state = next_state
        if done: break


    total_rewards.append(np.sum(rewards))
    average_reward = np.mean(total_rewards[-100:])

    total_steps.append(episode_steps)
    average_steps = np.mean(total_steps[-100:])

    episode_end_time = time.time()
    episode_duration = episode_end_time - episode_start_time
    if i == start_episode:
        training_start_time = episode_start_time
    total_training_time = episode_end_time - training_start_time
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    print(f"Ep {i}: Rtrn = {total_rewards[-1]:.2f} | ep steps = {episode_steps} | episode time = {format_time(episode_duration)} | total training time = {format_time(total_training_time)}")


    if Q_learning:

        #--------------------saving-------------------------
        if (i%5==0): 
            torch.save(algo.actor.state_dict(), 'actor_model.pt')
            torch.save(algo.critic.state_dict(), 'critic_model.pt')
            torch.save(algo.critic_target.state_dict(), 'critic_target_model.pt')
            #print("saving... len = ", len(replay_buffer))
            with open('replay_buffer', 'wb') as file:
                pickle.dump({'buffer': replay_buffer, 'x_coor':algo.actor.noise.x_coor, 'total_rewards':total_rewards, 'total_steps':total_steps, 'average_steps': average_steps}, file)
            #print(" > done")


        #-----------------validation-------------------------
        if (i>=start_test and i%50==0):
            if testing(env_test, limit_step=limit_eval, test_episodes=training_validation_episodes):
                print("Terminating training: testing returned True.")
                break
              

#====================================================
# * Apart from the algo core, fade_factor, tr_between_ep and limit_steps are crucial parameters for speed of training.
#   E.g. limit_steps = 700 instead of 2000 introduce less variance and makes BipedalWalkerHardcore's Agent less discouraged to go forward.
#   high values in tr_between_ep can make a "stiff" agent, but sometimes it is helpful for straight posture from the beginning (Humanoid-v4).


