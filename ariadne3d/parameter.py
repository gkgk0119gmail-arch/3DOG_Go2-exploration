# saving path
FOLDER_NAME = 'go2_turn_3d'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'

# save training data
SUMMARY_WINDOW = 32  # how many training steps before writing data to tensorboard
LOAD_MODEL = False  # do you want to load the model trained before
SAVE_IMG_GAP = 100  # how many episodes before saving a gif

# map and planning resolution
CELL_SIZE = 0.4  # meter, your map resolution
NODE_RESOLUTION = float(__import__('os').environ.get('ARIADNE_NODE_RES', 4.0))  # meter, your node resolution
FRONTIER_CELL_SIZE = 2 * CELL_SIZE  # do you want to downsample the frontiers

# map representation
FREE = 255  # value of free cells in the map
OCCUPIED = 1  # value of obstacle cells in the map
UNKNOWN = 127  # value of unknown cells in the map

# sensor and utility range
SENSOR_RANGE = 16  # meter
UTILITY_RANGE = 0.8 * SENSOR_RANGE  # consider frontiers within this range as observable
MIN_UTILITY = 2  # ignore the utility if observable frontiers are less than this value

# updating map range w.r.t the robot
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION  # nodes outside this range will not be affected by current measurements

# training parameters
MAX_EPISODE_STEP = int(__import__('os').environ.get('ARIADNE_MAX_STEP', 200))  # 2D exploration (<= 128 in the original) + time to finish the 3D scan
REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 128
LR = 1e-5
GAMMA = 1
NUM_META_AGENT = 20  # worker processes (32 threads here)

# network parameters
NODE_INPUT_DIM = 6  # original 4 (x, y, utility, guidepost) + heading cost + 3D utility
EMBEDDING_DIM = 128

# Graph parameters
K_SIZE = 25  # the number of neighboring nodes, fixed
NODE_PADDING_SIZE = int(__import__('os').environ.get('ARIADNE_NODE_PAD', 360))  # the number of nodes will be padded to this value, need it for batch training

# GPU usage
USE_GPU = False  # do you want to collect training data using GPUs (better not)
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 0  # 0 unless you want to collect data using GPUs


# --- 3DOG: Go2 turning cost + tilted VLP-16 3D coverage (env3d.py, lidar3d.py) ---------------------------------
PRETRAINED = 'pretrained/ariadne_ral2024.pth'  # RA-L 2024 checkpoint; new input weights start at zero
TILT_DEG = 15.0  # VLP-16 nose-up mount tilt (tools/lidar_mount_study.py: 15 deg is the sweet spot)
V_MAX = 1.0  # [m/s] walking policy forward speed
OMEGA = 1.5  # [rad/s] turn-in-place yaw rate
TURN_THR = 0.44  # [rad] heading error above which the Go2 stops and turns in place (turn_threshold)
TURN_SETTLE = 0.5  # [s] stop + restart per in-place turn
SCAN_STEP = 1.0  # [m] 3D sweep spacing while walking
SCAN_TURN = 0.785  # [rad] 3D sweep spacing while turning in place
W3D = 1.0  # weight of the 3D term
A_REF = 250.0  # [m^2] new wall + ceiling area worth reward 1 (calibrated: 3D term ~ frontier term per step)
STALL_AREA = 2.0  # [m^2] a step that adds less 3D area than this counts as a stall (after 2D completion)
STALL_STEPS = 4  # stalls in a row that end the episode
U3D_DONE = 40.0  # 3D utility (unseen elements in view) below which the 3D scan counts as finished
U3D_NORM = 1500.0  # 3D utility feature scale
UTIL3D = __import__('os').environ.get('ARIADNE_UTIL3D', 'view')  # 'view': ViewGain sweep per node (v2) | 'grid': annulus (v1)
U3D_NORM_VIEW = 200.0  # view gain feature scale [unseen elements hit by one sweep]
U3D_DONE_VIEW = 15.0  # 3D scan finished when no node's sweep would hit more unseen elements than this

# --- 3DOG step 2: follow the BIM-based scan plan (bim_env.py, bim_expert.py; --world bim only) ---------------
W_GUIDE = float(__import__('os').environ.get('ARIADNE_W_GUIDE', 0.0))  # weight of the guide term (driver3d --w_guide; 0 = off, eval)
GUIDE_ONLY = __import__('os').environ.get('ARIADNE_GUIDE_ONLY', '0') == '1'  # 1: guide term replaces the frontier / time / 3D terms (HEADER)
GUIDE_REACH = 0.75  # [x NODE_RESOLUTION] a plan point counts as passed when the robot comes this close in line of sight
