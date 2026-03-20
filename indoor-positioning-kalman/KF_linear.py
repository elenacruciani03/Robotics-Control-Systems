# KF implementation  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d

# Calculate Q matrix

static_points = {
    '27_33': '27_33-01_31-20_13_34.csv',
    '40_40': '40_40-01_31-20_00_03.csv',
    '40_65': '40_65-01_31-20_06_14.csv',
    '53_28': '53_28-01_31-20_08_44.csv'
}

Q_list = []

for name, file in static_points.items():
    df = pd.read_csv(file)
    x_raw = df['x_raw'].values
    y_raw = df['y_raw'].values
    
    mask = ~(np.isnan(x_raw) | np.isnan(y_raw))
    cov_matrix = np.cov(x_raw[mask], y_raw[mask], ddof=1)
    Q_list.append(cov_matrix)

Q = np.mean(Q_list, axis=0)

sigma_x = np.sqrt(Q[0,0])
sigma_y = np.sqrt(Q[1,1])

print(f"\Matrix Q:")
print(f"Q = [[{Q[0,0]:7.4f}, {Q[0,1]:7.4f}]")
print(f"     [{Q[1,0]:7.4f}, {Q[1,1]:7.4f}]]  cm²")
print(f"σ_x={np.sqrt(Q[0,0]):.3f} cm, σ_y={np.sqrt(Q[1,1]):.3f} cm")

# TRAJECTORIES ANALYSIS TO CALCULATE delta_t E sigma__a

trajectory_files = [
    'tr1_line-01_31-20_48_52.csv',
    'tr2_square-01_31-21_00_01.csv',
    'tr3_circle-01_31-21_04_24.csv',
    'tr4_circle-01_31-21_07_53.csv',
    'tr5_infinite-01_31-21_10_55.csv',
    'tr6_infinite-01_31-21_12_50.csv'
]

all_dt = [] 
dt_per_trajectory = [] 

for traj_file in trajectory_files:
    df = pd.read_csv(traj_file)
    timestamps = df['timestamp'].values
    
    dt_array = np.diff(timestamps)  
    all_dt.extend(dt_array)
    
    mean_dt = np.mean(dt_array) 
    dt_per_trajectory.append(mean_dt)
    
    traj_name = traj_file.split('-')[0] 
    print(f"{traj_name:15s}: N={len(timestamps):3d}, Δt={mean_dt:.4f} s")

delta_t = np.mean(all_dt) 
delta_t_std = np.std(all_dt, ddof=1) 

print(f"\nΔt (mean of all trajectories):")
print(f"Δt = {delta_t:.4f} ± {delta_t_std:.4f} s")
print(f"Sampling rate: {1/delta_t:.2f} Hz")
print(f"Variability between trajectories: {np.std(dt_per_trajectory, ddof=1):.4f} s")

# MATRICES A, C, R
# Matrix A

# State: x = [x, y, vx, vy]^T
# Model: x(k) = x(k-1) + vx(k-1)*Δt
#          y(k) = y(k-1) + vy(k-1)*Δt
#          vx(k) = vx(k-1)  
#          vy(k) = vy(k-1)  

A = np.array([
    [1, 0, delta_t, 0],
    [0, 1, 0, delta_t],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# Matrix C (Observation)

# We observe only the positions (x, y), not the velocities
C = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

# Matrix R (Process Noise)

# Parameters to be calibrated iteratively
#sigma_pos = 0.01  # cm - POSITION uncertainty in the model 
#sigma_vel = 0.3  # cm/s - VELOCITY uncertainty in the model
scala = 5e-3 # or 5e-3

# Diagonal matrix R 
pesi =  np.array([
    1.0,   
    1.0,   
    5.0, 
    5.0   
])

R = scala * np.diag(pesi**2)

# R = np.diag([
#    sigma_pos**2,  
#    sigma_pos**2,  
#    sigma_vel**2,  
#    sigma_vel**2   
# ])

print("\n")
print(f"\nQ/R = {Q[0,0]/R[0,0]:.1f}")

# Initial conditions (x_0, P_0)

# Initial state
x_0_template = np.array([0, 0, 0, 0])  # [x, y, vx, vy]

# Initial covariance
P_0 = np.diag([
    Q[0,0],   
    Q[1,1],   
    100.0,    
    100.0      
])


# KALMAN FILTER IMPLEMENTATION

def kalman_filter(z_raw, A, C, Q, R, P_0):

    N = len(z_raw)
    
    x_est_history = np.zeros((N, 4))
    P_history = []


    x_est = np.array([
        z_raw[0, 0],  
        z_raw[0, 1], 
        0.0,        
        0.0          
    ])
    
    P_est = P_0.copy()
    
    
    x_est_history[0] = x_est
    P_history.append(P_est.copy())

    
    for k in range(1, N):
        
        x_pred = A @ x_est  
        P_pred = A @ P_est @ A.T + R  
        z_k = z_raw[k] 
        z_pred = C @ x_pred  
        innovation = z_k - z_pred  

        # KALMAN GAIN
        S_k = C @ P_pred @ C.T + Q  
        K_k = P_pred @ C.T @ np.linalg.inv(S_k)  
        x_est = x_pred + K_k @ innovation  
        I = np.eye(4)
        P_est = (I - K_k @ C) @ P_pred 
        

        # SAVING RESULTS
        x_est_history[k] = x_est
        P_history.append(P_est.copy())
    
    return x_est_history, P_history


# VISUALIZATION AND GROUND TRUTH CALIBRATION


def calibrate_ground_truth_interactive(csv_file):
    
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_file)
    z_raw = df[['x_raw', 'y_raw']].values
    mask = ~(np.isnan(z_raw[:, 0]) | np.isnan(z_raw[:, 1]))
    z_raw_clean = z_raw[mask]

    anchors = np.array([
        [df['s1_x'].iloc[0], df['s1_y'].iloc[0]],
        [df['s2_x'].iloc[0], df['s2_y'].iloc[0]],
        [df['s3_x'].iloc[0], df['s3_y'].iloc[0]]
    ])
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-10, 90)
    ax.set_ylim(-10, 90)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0, 0), 80, 80, fill=False, 
                           edgecolor='black', linewidth=2))
    
    for i, (x, y) in enumerate(anchors):
        ax.plot(x, y, 'o', color='red', markersize=14, zorder=5)
        ax.text(x, y-5, f'S{i+1}', ha='center', va='top', 
                fontsize=10, fontweight='bold', color='red')
    
    ax.plot(z_raw_clean[:, 0], z_raw_clean[:, 1], 
            'o', color='lightblue', markersize=4, alpha=0.6, 
            label='Raw data', zorder=1)
    
    ax.plot(z_raw_clean[0, 0], z_raw_clean[0, 1], 
            'o', color='orange', markersize=12, label='Start')
    ax.plot(z_raw_clean[-1, 0], z_raw_clean[-1, 1], 
            's', color='purple', markersize=12, label='End')
    
    traj_name = csv_file.split('-')[0]
    ax.set_title(f'CALIBRAZIONE Ground Truth: {traj_name}\n'
                 f'CLICCA sui punti ideali, poi CHIUDI la finestra', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    points = []
    markers = []
    texts = []
    
    def onclick(event):
        if event.inaxes == ax and event.button == 1:  
            x, y = event.xdata, event.ydata
            points.append([x, y])
            
            marker, = ax.plot(x, y, 'o', color='red', markersize=12, 
                             markeredgecolor='black', markeredgewidth=2, zorder=10)
            text = ax.text(x+2, y+2, f'{len(points)-1}', 
                          fontsize=12, fontweight='bold', color='red', zorder=10)
            
            markers.append(marker)
            texts.append(text)
            
            fig.canvas.draw()
            print(f"Punto {len(points)-1}: [{x:.1f}, {y:.1f}]")
    
    def onkey(event):
        
        if event.key == 'u' and len(points) > 0:
            points.pop()
            markers[-1].remove()
            texts[-1].remove()
            markers.pop()
            texts.pop()
            fig.canvas.draw()
            
    
    cid1 = fig.canvas.mpl_connect('button_press_event', onclick)
    cid2 = fig.canvas.mpl_connect('key_press_event', onkey)
    
    plt.show()

    if len(points) > 0:
        print("\n" + "="*80)
        print(f"✓ Calibration completed: {len(points)} points")
        print("="*80)
        print(f"\n// COPY THIS into the ground_truth_points dictionary:\n")
        print(f"'{traj_name}': np.array([")
        for i, (x, y) in enumerate(points):
            print(f"    [{x:.1f}, {y:.1f}],  # Point {i}")
        print("]),\n")
        print("="*80 + "\n")
    else:
        print("\nNo points selected!\n")
    
    return np.array(points) if len(points) > 0 else None


def load_ground_truth(csv_file, num_points=200):
    
    ground_truth_points = {
        
        'tr1_line': np.array([
            [41.8, 28],   # Point START 
            [41.7, 35],   # Point intermediate 1
            [41.6, 45],   # Point intermediate 2
            [41.5, 55],   # Point intermediate 3
            [41.5, 65],   # Point intermediate 4
            [41.5, 72]    # Point END 
        ]),
        
        'tr2_square': np.array([
            [42.8, 26.5],  # Point 0
            [26.8, 40.1],  # Point 1
            [41.3, 56.7],  # Point 2
            [58.0, 40.5],  # Point 3
            [43.4, 26.5],  # Point 4
        ]),
        
        'tr3_circle': np.array([
            [42.3, 29.9],  # Point 0
            [40.6, 30.0],  # Point 1
            [38.9, 30.9],  # Point 2
            [37.5, 32.0],  # Point 3
            [35.8, 33.5],  # Point 4
            [34.5, 34.9],  # Point 5
            [33.3, 36.4],  # Point 6
            [32.2, 37.9],  # Point 7
            [31.9, 39.6],  # Point 8
            [32.1, 41.5],  # Point 9
            [32.7, 43.1],  # Point 10
            [33.6, 44.9],  # Point 11
            [35.3, 46.7],  # Point 12
            [36.4, 47.9],  # Point 13
            [37.9, 49.0],  # Point 14
            [39.8, 50.2],  # Point 15
            [40.6, 50.6],  # Point 16
            [41.8, 50.7],  # Point 17
            [43.1, 50.6],  # Point 18
            [44.3, 50.0],  # Point 19
            [45.4, 49.2],  # Point 20
            [46.6, 48.3],  # Point 21
            [47.8, 47.5],  # Point 22
            [49.0, 46.5],  # Point 23
            [49.6, 45.5],  # Point 24
            [50.6, 44.2],  # Point 25
            [51.2, 43.0],  # Point 26
            [51.4, 41.7],  # Point 27
            [51.6, 40.4],  # Point 28
            [51.4, 38.7],  # Point 29
            [50.7, 37.0],  # Point 30
            [50.1, 35.7],  # Point 31
            [49.3, 34.6],  # Point 32
            [48.3, 33.4],  # Point 33
            [47.1, 32.4],  # Point 34
            [45.7, 31.5],  # Point 35
            [44.2, 30.6],  # Point 36
            [43.0, 30.0],  # Point 37
        ]),

        'tr4_circle': np.array([
            [41.7, 30.7],  # Point 0
            [40.9, 30.7],  # Point 1
            [39.8, 31.3],  # Point 2
            [38.8, 32.0],  # Point 3
            [38.0, 32.9],  # Point 4
            [37.6, 33.7],  # Point 5
            [36.7, 34.5],  # Point 6
            [35.8, 35.8],  # Point 7
            [35.1, 37.1],  # Point 8
            [34.4, 38.2],  # Point 9
            [33.9, 39.2],  # Point 10
            [33.3, 40.6],  # Point 11
            [33.3, 41.9],  # Point 12
            [33.7, 43.2],  # Point 13
            [34.3, 44.5],  # Point 14
            [35.5, 45.7],  # Point 15
            [36.5, 46.5],  # Point 16
            [37.3, 47.1],  # Point 17
            [38.2, 47.7],  # Point 18
            [39.6, 48.3],  # Point 19
            [40.6, 48.7],  # Point 20
            [41.8, 48.7],  # Point 21
            [43.0, 48.7],  # Point 22
            [44.1, 48.4],  # Point 23
            [45.4, 48.2],  # Point 24
            [46.0, 47.6],  # Point 25
            [46.5, 47.1],  # Point 26
            [47.2, 46.5],  # Point 27
            [48.1, 46.0],  # Point 28
            [48.4, 45.4],  # Point 29
            [49.0, 44.8],  # Point 30
            [49.8, 44.1],  # Point 31
            [50.4, 43.0],  # Point 32
            [50.7, 42.2],  # Point 33
            [50.9, 41.6],  # Point 34
            [51.1, 40.6],  # Point 35
            [51.0, 39.5],  # Point 36
            [51.1, 38.8],  # Point 37
            [51.0, 37.5],  # Point 38
            [50.9, 36.8],  # Point 39
            [50.4, 35.9],  # Point 40
            [50.0, 35.2],  # Point 41
            [49.2, 34.6],  # Point 42
            [48.6, 34.0],  # Point 43
            [48.1, 33.0],  # Point 44
            [47.1, 32.4],  # Point 45
            [46.0, 31.9],  # Point 46
            [45.4, 31.3],  # Point 47
            [44.3, 30.8],  # Point 48
            [43.2, 30.4],  # Point 49
        ]),
        
        'tr5_infinite': np.array([
            [42.2, 47.1],  # Point 0
            [41.5, 48.1],  # Point 1
            [40.3, 49.5],  # Point 2
            [39.4, 50.8],  # Point 3
            [38.8, 52.2],  # Point 4
            [38.2, 53.8],  # Point 5
            [37.2, 56.1],  # Point 6
            [37.2, 57.5],  # Point 7
            [37.0, 59.0],  # Point 8
            [37.3, 60.7],  # Point 9
            [37.8, 62.0],  # Point 10
            [38.9, 63.2],  # Point 11
            [40.1, 64.0],  # Point 12
            [41.5, 64.0],  # Point 13
            [42.3, 63.7],  # Point 14
            [43.2, 63.2],  # Point 15
            [44.1, 62.1],  # Point 16
            [44.8, 61.3],  # Point 17
            [45.5, 60.1],  # Point 18
            [45.9, 59.0],  # Point 19
            [46.0, 57.6],  # Point 20
            [45.8, 56.4],  # Point 21
            [45.5, 54.9],  # Point 22
            [45.0, 53.9],  # Point 23
            [44.8, 52.2],  # Point 24
            [44.0, 50.8],  # Point 25
            [43.0, 49.5],  # Point 26
            [42.5, 48.5],  # Point 27
            [42.0, 47.5],  # Point 28
            [41.2, 45.8],  # Point 29
            [40.7, 45.1],  # Point 30
            [40.2, 44.1],  # Point 31
            [39.6, 43.0],  # Point 32
            [39.3, 41.9],  # Point 33
            [38.6, 40.6],  # Point 34
            [38.3, 39.4],  # Point 35
            [38.0, 37.9],  # Point 36
            [37.7, 36.5],  # Point 37
            [37.0, 34.7],  # Point 38
            [36.9, 32.5],  # Point 39
            [37.8, 31.4],  # Point 40
            [39.4, 30.9],  # Point 41
            [40.9, 30.7],  # Point 42
            [42.5, 30.5],  # Point 43
            [44.5, 30.5],  # Point 44
            [45.3, 30.7],  # Point 45
            [46.5, 31.4],  # Point 46
            [47.6, 32.4],  # Point 47
            [48.4, 33.5],  # Point 48
            [49.0, 34.9],  # Point 49
            [48.9, 36.4],  # Point 50
            [48.5, 37.8],  # Point 51
            [48.1, 39.5],  # Point 52
            [47.3, 41.1],  # Point 53
            [46.2, 42.1],  # Point 54
            [45.0, 43.7],  # Point 55
            [44.1, 44.7],  # Point 56
            [43.0, 46.0],  # Point 57
            [42.2, 46.9],  # Point 58
        ]),
        
        'tr6_infinite': np.array([
            [41.8, 39.2],  # Point 0
            [40.4, 39.7],  # Point 1
            [40.0, 40.4],  # Point 2
            [39.3, 40.8],  # Point 3
            [38.5, 41.5],  # Point 4
            [37.7, 42.3],  # Point 5
            [36.6, 43.0],  # Point 6
            [35.5, 43.5],  # Point 7
            [34.2, 44.0],  # Point 8
            [32.5, 44.3],  # Point 9
            [31.2, 43.9],  # Point 10
            [30.0, 43.1],  # Point 11
            [29.3, 42.2],  # Point 12
            [28.8, 41.4],  # Point 13
            [28.6, 40.3],  # Point 14
            [28.0, 39.1],  # Point 15
            [28.0, 38.0],  # Point 16
            [28.3, 37.2],  # Point 17
            [28.9, 36.8],  # Point 18
            [30.3, 36.4],  # Point 19
            [31.7, 36.4],  # Point 20
            [32.7, 36.3],  # Point 21
            [34.1, 36.4],  # Point 22
            [35.1, 36.6],  # Point 23
            [36.5, 36.9],  # Point 24
            [37.5, 37.3],  # Point 25
            [38.4, 37.6],  # Point 26
            [39.6, 38.2],  # Point 27
            [40.5, 38.4],  # Point 28
            [41.5, 38.9],  # Point 29
            [42.8, 39.7],  # Point 30
            [44.1, 40.4],  # Point 31
            [45.0, 41.0],  # Point 32
            [46.1, 41.7],  # Point 33
            [47.3, 42.3],  # Point 34
            [48.5, 42.9],  # Point 35
            [49.6, 43.4],  # Point 36
            [50.7, 43.4],  # Point 37
            [52.5, 43.5],  # Point 38
            [53.7, 43.2],  # Point 39
            [54.6, 42.8],  # Point 40
            [55.4, 41.9],  # Point 41
            [55.9, 41.3],  # Point 42
            [56.0, 40.2],  # Point 43
            [55.8, 39.1],  # Point 44
            [55.6, 37.7],  # Point 45
            [55.3, 37.0],  # Point 46
            [54.8, 36.2],  # Point 47
            [53.7, 35.2],  # Point 48
            [52.8, 34.7],  # Point 49
            [51.4, 34.3],  # Point 50
            [50.1, 34.3],  # Point 51
            [49.0, 34.5],  # Point 52
            [47.6, 34.9],  # Point 53
            [46.6, 35.8],  # Point 54
            [45.4, 36.8],  # Point 55
            [44.4, 37.7],  # Point 56
            [43.0, 38.6],  # Point 57
        ])
    }
    
    traj_name = csv_file.split('-')[0] 
    traj_type = csv_file.split('_')[1].split('-')[0]

    if traj_name not in ground_truth_points:
        print(f"Ground truth non disponibile per {traj_name}")
        return None
    
    points = ground_truth_points[traj_name]
    n_keypoints = len(points)

    if traj_type == 'square':
        
        if n_keypoints < 4:
            print(f"ERROR: Square requires at least 4 vertices, found {n_keypoints}")
            return None
        
        n_sides = n_keypoints - 1
        points_per_side = num_points // n_sides
        
        x_gt = []
        y_gt = []
        
        for i in range(n_sides):
            t_seg = np.linspace(0, 1, points_per_side)
            x_seg = points[i, 0] + t_seg * (points[i+1, 0] - points[i, 0])
            y_seg = points[i, 1] + t_seg * (points[i+1, 1] - points[i, 1])
            
            x_gt.extend(x_seg)
            y_gt.extend(y_seg)
        
        x_gt = np.array(x_gt[:num_points])
        y_gt = np.array(y_gt[:num_points])
        
        return np.column_stack([x_gt, y_gt])
    
    t = np.linspace(0, 1, n_keypoints)
    t_new = np.linspace(0, 1, num_points)
    
    try:
        fx = interp1d(t, points[:, 0], kind='cubic', fill_value='extrapolate')
        fy = interp1d(t, points[:, 1], kind='cubic', fill_value='extrapolate')
        
        x_gt = fx(t_new)
        y_gt = fy(t_new)
        
        z_gt = np.column_stack([x_gt, y_gt])
        
        return z_gt
    
    except Exception as e:
        print(f"Errore interpolazione ground truth: {e}")
        return None


def plot_trajectory(csv_file, title=None, save_png=True, show_plot=True):
    
    print(f"\n{'='*80}")
    print(f"VISUALIZATION: {csv_file}")
    print(f"{'='*80}\n")
    
    df = pd.read_csv(csv_file)
    z_raw = df[['x_raw', 'y_raw']].values
    mask = ~(np.isnan(z_raw[:, 0]) | np.isnan(z_raw[:, 1]))
    z_raw_clean = z_raw[mask]
    
    z_gt = load_ground_truth(csv_file, num_points=len(z_raw_clean))
    
    anchors = np.array([
        [df['s1_x'].iloc[0], df['s1_y'].iloc[0]],
        [df['s2_x'].iloc[0], df['s2_y'].iloc[0]],
        [df['s3_x'].iloc[0], df['s3_y'].iloc[0]]
    ])
    
    print(f"Samples: {len(z_raw_clean)} | Sensors: S1={anchors[0]}, S2={anchors[1]}, S3={anchors[2]}")
    
    x_filtered, P_hist = kalman_filter(z_raw_clean, A, C, Q, R, P_0)
    x_filt, y_filt = x_filtered[:, 0], x_filtered[:, 1]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if title is None:
        title = csv_file.split('-')[0].replace('_', ' ').title()
    fig.canvas.manager.set_window_title(f'Kalman Filter - {title}')
    
    ax.set_xlim(-10, 90)
    ax.set_ylim(-10, 90)
    ax.set_xlabel('X (cm)', fontsize=12)
    ax.set_ylabel('Y (cm)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    ax.add_patch(Rectangle((0, 0), 80, 80, fill=False, 
                           edgecolor='black', linewidth=2))
    
    for i, (x, y) in enumerate(anchors):
        ax.plot(x, y, 'o', color='red', markersize=14, 
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        ax.text(x, y-5, f'S{i+1}', ha='center', va='top', 
                fontsize=10, fontweight='bold', color='red')
    
    
    # 1. Ground Truth 
    if z_gt is not None:
        ax.plot(z_gt[:, 0], z_gt[:, 1], 
                ':', color="#006AFF", linewidth=2.5, alpha=0.7, 
                label='True Position', zorder=2)
    
    # 2. Raw 
    ax.plot(z_raw_clean[:, 0], z_raw_clean[:, 1], 
            '--', color="#DA2020", linewidth=2, alpha=0.6, 
            label='Trilateration', zorder=3)
    
    # 3. Filtered (Kalman)
    ax.plot(x_filt, y_filt, 
            '-', color="#009F42", linewidth=2, alpha=1, 
            label='Kalman Filtered', zorder=4)
    
    ax.plot(z_raw_clean[0, 0], z_raw_clean[0, 1], 
            'o', color='orange', markersize=10, 
            markeredgecolor='black', markeredgewidth=1, 
            label='Start', zorder=6)
    ax.plot(x_filt[-1], y_filt[-1], 
            'o', color='purple', markersize=10, 
            markeredgecolor='black', markeredgewidth=1, 
            label='Finish', zorder=6)
    
    ax.set_title(f'Kalman Filter - {title}', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', fontsize=11, frameon=True, 
              shadow=True, fancybox=True)
    
    
    plt.tight_layout()
    
    if save_png:
        output_file = csv_file.replace('.csv', '_kalman.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    print(f"{'='*80}\n")
    
    return fig, ax


plot_trajectory('tr1_line-01_31-20_48_52.csv', title='Line', save_png=False)
plot_trajectory('tr2_square-01_31-21_00_01.csv', title='Square', save_png=False)
plot_trajectory('tr3_circle-01_31-21_04_24.csv', title='Circle', save_png=False)
plot_trajectory('tr4_circle-01_31-21_07_53.csv', title='Circle', save_png=False)
plot_trajectory('tr5_infinite-01_31-21_10_55.csv', title='Infinite', save_png=False)
plot_trajectory('tr6_infinite-01_31-21_12_50.csv', title='Infinite', save_png=False)

