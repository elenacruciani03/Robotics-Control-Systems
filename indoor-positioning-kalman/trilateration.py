import serial
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from datetime import datetime

# Configuration
ANCHORS = np.array([[0, 0], [80, 0], [40, 80]])
SERIAL_PORT = '/dev/cu.usbmodem1301'
BAUD_RATE = 115200

def trilateration(distances, anchors):
    if np.any(distances <= 0):
        return None
    
    x1, y1 = anchors[0]
    x2, y2 = anchors[1]
    x3, y3 = anchors[2]
    r1, r2, r3 = distances
    
    A = np.array([
        [2*(x3-x1), 2*(y3-y1)],
        [2*(x3-x2), 2*(y3-y2)]
    ])
    
    b = np.array([
        r1**2 - r3**2 - x1**2 - y1**2 + x3**2 + y3**2,
        r2**2 - r3**2 - x2**2 - y2**2 + x3**2 + y3**2
    ])
    
    try:
        if abs(np.linalg.det(A)) < 1e-10:
            return None
        return np.linalg.solve(A, b)
    except:
        return None

# Setup plot
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

ax.set_xlim(0, 80)
ax.set_ylim(0, 80)
ax.set_aspect('equal')
ax.set_xlabel('X (cm)', fontsize=12)
ax.set_ylabel('Y (cm)', fontsize=12)
ax.grid(True, alpha=0.3)

# Measurement area
ax.add_patch(plt.Rectangle((0, 0), 80, 80, fill=False, 
                           edgecolor='black', linewidth=2))

# Sensors
for i, (x, y) in enumerate(ANCHORS):
    ax.plot(x, y, 'o', color='red', markersize=14, 
            markeredgecolor='black', markeredgewidth=1.5, zorder=5)

# Object point
point, = ax.plot([], [], 'o', color='blue', 
                markersize=15, markeredgecolor='black',
                markeredgewidth=1.5, zorder=10)

# Trajectory
trajectory, = ax.plot([], [], '-', color='blue', 
                     linewidth=2, alpha=0.5, zorder=8)

# Title
ax.set_title('RTLS - Data Acquisition', fontsize=14, 
             fontweight='bold', loc='left', pad=20)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
           markersize=10, markeredgecolor='black', label='Sensors'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
           markersize=12, markeredgecolor='black', label='Object'),
    Line2D([0], [0], color='blue', linewidth=2, alpha=0.5, label='Path')
]

ax.legend(handles=legend_elements, loc='lower right', 
          bbox_to_anchor=(0.98, 1.02), ncol=3, fontsize=10, 
          frameon=True, borderaxespad=0)

# Position text
text_pos = ax.text(0.5, -0.12, '', ha='center', va='top',
                   fontsize=12, fontweight='bold',
                   transform=ax.transAxes)

plt.tight_layout()

# Data storage
data_log = []
x_history = []
y_history = []
start_time = time.time()

# Serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

print("Acquisition Started | Close window to stop")
print(f"S1={ANCHORS[0]} S2={ANCHORS[1]} S3={ANCHORS[2]}\n")

sample_count = 0

try:
    while plt.fignum_exists(fig.number):
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            
            try:
                distances = np.array([float(x) for x in line.split(',')])
                
                if len(distances) == 3:
                    pos = trilateration(distances, ANCHORS)
                    
                    if pos is not None:
                        x, y = pos
                        current_time = time.time() - start_time
                        
                        # Store data with anchor coordinates
                        data_log.append({
                            'timestamp': current_time,
                            'sample': sample_count,
                            'd1': distances[0],
                            'd2': distances[1],
                            'd3': distances[2],
                            'x_raw': x,
                            'y_raw': y,
                            's1_x': ANCHORS[0, 0],
                            's1_y': ANCHORS[0, 1],
                            's2_x': ANCHORS[1, 0],
                            's2_y': ANCHORS[1, 1],
                            's3_x': ANCHORS[2, 0],
                            's3_y': ANCHORS[2, 1]
                        })
                        
                        # Update trajectory
                        x_history.append(x)
                        y_history.append(y)
                        
                        sample_count += 1
                        
                        # Update plot
                        point.set_data([x], [y])
                        trajectory.set_data(x_history, y_history)
                        text_pos.set_text(f'Position: ({x:.1f}, {y:.1f}) cm | Samples: {sample_count}')
                        
                        # Redraw
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                        
                        # Compact terminal output every 10 samples
                        if sample_count % 10 == 0:
                            print(f"[{sample_count:4d}] {current_time:5.1f}s | ({x:5.1f}, {y:5.1f}) cm")
                        
            except:
                pass

except KeyboardInterrupt:
    print("\nStopped\n")

finally:
    ser.close()
    plt.ioff()
    
    # Save data
    if len(data_log) > 0:
        df = pd.DataFrame(data_log)
        
        # Filename with timestamp
        timestamp_str = datetime.now().strftime("%m_%d-%H_%M_%S")
        filename = f'prova-{timestamp_str}.csv'
        
        df.to_csv(filename, index=False)
        
        # Calculate statistics
        duration = df['timestamp'].max()
        avg_freq = len(df) / duration if duration > 0 else 0  
    
        print(f"Saved: {filename}")
        print(f"Samples: {len(df)} | Duration: {duration:.1f}s | Freq: {avg_freq:.1f} Hz")
    else:
        print("No data acquired")
    
    plt.show()