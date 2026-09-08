"""
Generate synthetic drone-like video for testing ARGUS pipeline
Creates a moving scene with plenty of SIFT features
"""

import cv2
import numpy as np
import os
import argparse
import tempfile

def generate_synthetic_video(output_path, num_frames=150, width=1280, height=720, fps=30):
    """
    Generate synthetic video that simulates drone forward motion over textured terrain
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    np.random.seed(42)
    # Pre-generate random features (buildings, roads, etc.)
    # Create a large virtual world
    world_w, world_h = width*3, height*3
    # Random colored squares and circles as features
    features = []
    for _ in range(300):
        x = np.random.randint(0, world_w)
        y = np.random.randint(0, world_h)
        size = np.random.randint(20, 80)
        color = (np.random.randint(50,255), np.random.randint(50,255), np.random.randint(50,255))
        shape_type = np.random.choice(['rect','circle','line'])
        features.append((x,y,size,color,shape_type))

    print(f"Generating {num_frames} frames to {output_path}...")
    for i in range(num_frames):
        # Simulate forward motion + slight lateral drift
        offset_x = int(i * 8)  # forward
        offset_y = int(np.sin(i*0.05)*20)  # slight sinusoidal drift like drone

        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Background gradient (sky to ground)
        for y in range(height):
            # Gradient from light blue sky to greenish ground
            t = y / height
            b = int(135 + t*50)
            g = int(206 - t*30)
            r = int(235 - t*50)
            img[y, :] = (b, g, r)

        # Draw world features shifted by offset
        for (wx, wy, size, color, stype) in features:
            cx = wx - offset_x
            cy = wy - offset_y
            # Only draw if in view (+ margin)
            if -size <= cx <= width+size and -size <= cy <= height+size:
                if stype == 'rect':
                    cv2.rectangle(img, (cx, cy), (cx+size, cy+size), color, -1)
                    cv2.rectangle(img, (cx, cy), (cx+size, cy+size), (0,0,0), 2)
                elif stype == 'circle':
                    cv2.circle(img, (cx+size//2, cy+size//2), size//2, color, -1)
                    cv2.circle(img, (cx+size//2, cy+size//2), size//2, (0,0,0), 2)
                else:
                    cv2.line(img, (cx, cy), (cx+size, cy+size), color, 3)

        # Add some larger structures that persist (buildings)
        # Building 1
        b1_x = 400 - offset_x//2
        if -200 <= b1_x <= width:
            cv2.rectangle(img, (b1_x, 200), (b1_x+150, 500), (200,200,200), -1)
            cv2.rectangle(img, (b1_x, 200), (b1_x+150, 500), (0,0,0), 3)
            # Windows
            for wy in range(220, 480, 40):
                for wx in range(b1_x+10, b1_x+130, 30):
                    cv2.rectangle(img, (wx, wy), (wx+15, wy+20), (255,255,0), -1)

        # Road
        road_y = height//2 + offset_y//2
        cv2.rectangle(img, (0, road_y), (width, road_y+80), (50,50,50), -1)
        # Dashed line
        for x in range(0, width, 60):
            cv2.rectangle(img, (x, road_y+35), (x+30, road_y+45), (255,255,255), -1)

        # Add frame number and simulated telemetry
        cv2.putText(img, f"ARGUS SIM | FRAME {i:04d} | ALT 50m | SPD 5m/s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
        cv2.putText(img, f"ARGUS SIM | FRAME {i:04d} | ALT 50m | SPD 5m/s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        out.write(img)

        if i % 30 == 0:
            print(f"  {i}/{num_frames} frames")

    out.release()
    print(f"Done: {output_path}, {os.path.getsize(output_path)/1024/1024:.2f} MB")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="Output mp4 path")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    if args.output is None:
        tmp = tempfile.gettempdir()
        out_path = os.path.join(tmp, "argus_synthetic.mp4")
    else:
        out_path = args.output

    generate_synthetic_video(out_path, num_frames=args.frames, width=args.width, height=args.height)
    print(f"\nTest with: python test_pipeline.py --video {out_path}")
    print(f"Or upload to Streamlit app")
