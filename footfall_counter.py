"""
Footfall Counter using Computer Vision - Escalator Mode
Author: Kaushal Kumar Thakur
Date: November 2025

Based on proven escalator counting techniques with separate entry/exit lines.
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import argparse
from pathlib import Path
import time
import json
import os


class ROISelector:
    """Interactive dual ROI selector for separate entry and exit lines."""
    
    def __init__(self):
        self.entry_points = []
        self.exit_points = []
        self.current_image = None
        self.current_mode = 'entry'
        self.window_name = "Dual ROI Selection"
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_mode == 'entry':
                self.entry_points.append((x, y))
                cv2.circle(self.current_image, (x, y), 7, (0, 255, 0), -1)
                if len(self.entry_points) > 1:
                    cv2.line(self.current_image, self.entry_points[-2], self.entry_points[-1], (0, 255, 0), 4)
            else:
                self.exit_points.append((x, y))
                cv2.circle(self.current_image, (x, y), 7, (0, 0, 255), -1)
                if len(self.exit_points) > 1:
                    cv2.line(self.current_image, self.exit_points[-2], self.exit_points[-1], (0, 0, 255), 4)
            cv2.imshow(self.window_name, self.current_image)
    
    def select_roi(self, frame):
        self.current_image = frame.copy()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n" + "="*70)
        print("ESCALATOR MODE - DUAL ROI SELECTION")
        print("="*70)
        print("STEP 1: Draw ENTRY LINE (GREEN)")
        print("  - Click 2 points horizontally across entry area")
        print("  - Press 'n' when done\n")
        print("STEP 2: Draw EXIT LINE (RED)")
        print("  - Click 2 points horizontally across exit area")
        print("  - Press 's' to SAVE\n")
        print("Controls: n=Next | r=Reset | s=Save | q=Quit")
        print("="*70 + "\n")
        
        while True:
            if self.current_mode == 'entry':
                title = "STEP 1: ENTRY LINE (GREEN) - Press 'n'"
            else:
                title = "STEP 2: EXIT LINE (RED) - Press 's'"
            cv2.setWindowTitle(self.window_name, title)
            cv2.imshow(self.window_name, self.current_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('n') and self.current_mode == 'entry' and len(self.entry_points) >= 2:
                self.current_mode = 'exit'
                print("✓ Entry line saved")
            elif key == ord('s') and len(self.entry_points) >= 2 and len(self.exit_points) >= 2:
                cv2.destroyAllWindows()
                print(f"\n✓ ROI Saved!")
                return {'entry': self.entry_points, 'exit': self.exit_points}
            elif key == ord('r'):
                if self.current_mode == 'entry':
                    self.entry_points = []
                else:
                    self.exit_points = []
                self.current_image = frame.copy()
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return None
    
    def save_roi_to_file(self, roi_config, filename='roi_config.json'):
        with open(filename, 'w') as f:
            json.dump({'roi_config': roi_config, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
        print(f"✓ Saved to: {filename}")
    
    def load_roi_from_file(self, filename='roi_config.json'):
        try:
            with open(filename, 'r') as f:
                return json.load(f)['roi_config']
        except:
            return None


class CentroidTracker:
    """Simple centroid tracker."""
    
    def __init__(self, max_disappeared=50, max_distance=80):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid):
        self.objects[self.next_object_id] = tuple(centroid)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, detections):
        if len(detections) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects
        
        centroids = np.array([(int((x1+x2)/2), int((y1+y2)/2)) for x1,y1,x2,y2 in detections])
        
        if len(self.objects) == 0:
            for c in centroids:
                self.register(c)
        else:
            oids = list(self.objects.keys())
            obj_centroids = np.array(list(self.objects.values()))
            
            D = np.linalg.norm(obj_centroids[:, np.newaxis] - centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows, used_cols = set(), set()
            
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols or D[row, col] > self.max_distance:
                    continue
                self.objects[oids[row]] = tuple(centroids[col])
                self.disappeared[oids[row]] = 0
                used_rows.add(row)
                used_cols.add(col)
            
            for row in set(range(len(obj_centroids))) - used_rows:
                self.disappeared[oids[row]] += 1
                if self.disappeared[oids[row]] > self.max_disappeared:
                    self.deregister(oids[row])
            
            for col in set(range(len(centroids))) - used_cols:
                self.register(centroids[col])
        
        return self.objects


class FootfallCounter:
    """Escalator-style footfall counter with zone detection."""
    
    def __init__(self, model_path='yolov8n.pt', roi_config=None, confidence_threshold=0.4, webcam=False):
        print(f"[INFO] Loading YOLOv8: {model_path}")
        self.model = YOLO(model_path)
        self.tracker = CentroidTracker()
        
        self.entry_line = roi_config.get('entry') if roi_config else None
        self.exit_line = roi_config.get('exit') if roi_config else None
        
        self.confidence_threshold = confidence_threshold
        self.webcam = webcam
        
        # ESCALATOR-STYLE COUNTING (like the reference code)
        self.entry_count_list = []  # List of IDs that crossed entry
        self.exit_count_list = []   # List of IDs that crossed exit
        
        # Tolerance zone (±pixels around line)
        self.tolerance = 15
        
        self.fps_values = deque(maxlen=30)
        
        print(f"[INFO] Escalator Mode - Tolerance: ±{self.tolerance}px")
        
    def check_line_crossing(self, object_id, cx, cy):
        """
        Escalator-style crossing detection.
        Checks if centroid (cx, cy) is within tolerance zone of either line.
        """
        # CHECK ENTRY LINE
        if self.entry_line and len(self.entry_line) >= 2:
            x1, y1 = self.entry_line[0]
            x2, y2 = self.entry_line[-1]
            
            # Check if centroid is within the line's X range and Y tolerance
            if x1 < cx < x2 and (y1 - self.tolerance) < cy < (y1 + self.tolerance):
                if object_id not in self.entry_count_list:
                    self.entry_count_list.append(object_id)
                    print(f"✓ [ENTRY] ID {object_id} crossed entry line")
                    return 'entry'
        
        # CHECK EXIT LINE
        if self.exit_line and len(self.exit_line) >= 2:
            x1, y1 = self.exit_line[0]
            x2, y2 = self.exit_line[-1]
            
            # Check if centroid is within the line's X range and Y tolerance
            if x1 < cx < x2 and (y1 - self.tolerance) < cy < (y1 + self.tolerance):
                if object_id not in self.exit_count_list:
                    self.exit_count_list.append(object_id)
                    print(f"✓ [EXIT] ID {object_id} crossed exit line")
                    return 'exit'
        
        return None
    
    def draw_visualizations(self, frame, detections, objects):
        h, w = frame.shape[:2]
        
        # Draw ENTRY LINE (GREEN)
        if self.entry_line and len(self.entry_line) >= 2:
            cv2.line(frame, self.entry_line[0], self.entry_line[-1], (0, 255, 0), 5)
            for pt in self.entry_line:
                cv2.circle(frame, pt, 8, (0, 255, 0), -1)
            
            mid_x = (self.entry_line[0][0] + self.entry_line[-1][0]) // 2
            mid_y = (self.entry_line[0][1] + self.entry_line[-1][1]) // 2
            cv2.putText(frame, "ENTRY", (mid_x-40, mid_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw EXIT LINE (RED)
        if self.exit_line and len(self.exit_line) >= 2:
            cv2.line(frame, self.exit_line[0], self.exit_line[-1], (0, 0, 255), 5)
            for pt in self.exit_line:
                cv2.circle(frame, pt, 8, (0, 0, 255), -1)
            
            mid_x = (self.exit_line[0][0] + self.exit_line[-1][0]) // 2
            mid_y = (self.exit_line[0][1] + self.exit_line[-1][1]) // 2
            cv2.putText(frame, "EXIT", (mid_x-30, mid_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw objects
        for oid, centroid in objects.items():
            cx, cy = centroid
            
            # Find bounding box
            for x1, y1, x2, y2 in detections:
                det_cx, det_cy = int((x1+x2)/2), int((y1+y2)/2)
                if (det_cx, det_cy) == (cx, cy):
                    # Color based on status
                    if oid in self.entry_count_list:
                        color = (0, 255, 0)
                        status = "ENTERED"
                    elif oid in self.exit_count_list:
                        color = (0, 0, 255)
                        status = "EXITED"
                    else:
                        color = (255, 165, 0)
                        status = "TRACKING"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.circle(frame, (cx, cy), 6, color, -1)
                    
                    label = f"ID:{oid} {status}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1-lh-10), (x1+lw, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    break
        
        # Count panel
        panel = np.zeros((150, w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        cv2.putText(panel, "ESCALATOR COUNTER", (w//2-160, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        entry_count = len(self.entry_count_list)
        exit_count = len(self.exit_count_list)
        
        cv2.putText(panel, f"ENTRIES: {entry_count}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        cv2.putText(panel, f"EXITS: {exit_count}", (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
        
        total = entry_count + exit_count
        inside = entry_count - exit_count
        
        cv2.putText(panel, f"TOTAL: {total}", (w-300, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        cv2.putText(panel, f"INSIDE: {inside}", (w-300, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 0), 3)
        
        return np.vstack([frame, panel])
    
    def process_video(self, video_source, output_path=None, show_realtime=True):
        cap = cv2.VideoCapture(0 if self.webcam else video_source)
        
        if not cap.isOpened():
            print("[ERROR] Cannot open video")
            return
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if not self.webcam else 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.webcam else 0
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h+150))
        
        frame_count = 0
        start = time.time()
        
        print("\n[INFO] Processing...\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            t1 = time.time()
            
            results = self.model(frame, classes=[0], conf=self.confidence_threshold, verbose=False)
            
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((x1, y1, x2, y2))
            
            objects = self.tracker.update(detections)
            
            # Check crossings for each tracked object
            for oid, (cx, cy) in objects.items():
                self.check_line_crossing(oid, cx, cy)
            
            output = self.draw_visualizations(frame, detections, objects)
            
            t2 = time.time()
            self.fps_values.append(1.0/(t2-t1) if t2>t1 else 0)
            avg_fps = np.mean(self.fps_values)
            
            cv2.putText(output, f"FPS: {avg_fps:.1f}", (w-150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if not self.webcam and total > 0:
                print(f"\r[{frame_count/total*100:5.1f}%] Entries:{len(self.entry_count_list)} Exits:{len(self.exit_count_list)}", end='')
            
            if show_realtime:
                cv2.imshow('Escalator Counter', output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if writer:
                writer.write(output)
        
        elapsed = time.time() - start
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print("\n\n" + "="*70)
        print("FINAL RESULTS - ESCALATOR MODE")
        print("="*70)
        print(f"Frames: {frame_count} | Time: {elapsed:.2f}s | FPS: {frame_count/elapsed:.2f}")
        print(f"✓ ENTRIES: {len(self.entry_count_list)}")
        print(f"✓ EXITS: {len(self.exit_count_list)}")
        print(f"✓ TOTAL: {len(self.entry_count_list) + len(self.exit_count_list)}")
        print(f"✓ INSIDE: {len(self.entry_count_list) - len(self.exit_count_list)}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Escalator Counter')
    parser.add_argument('--video', type=str)
    parser.add_argument('--webcam', action='store_true')
    parser.add_argument('--output', type=str, default='output/counted.mp4')
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--select-roi', action='store_true')
    parser.add_argument('--load-roi', type=str)
    parser.add_argument('--save-roi', type=str, default='roi_config.json')
    parser.add_argument('--confidence', type=float, default=0.4)
    parser.add_argument('--no-display', action='store_true')
    
    args = parser.parse_args()
    
    if not args.webcam and not args.video:
        parser.error("Need --video or --webcam")
    
    if args.video and not os.path.exists(args.video):
        print(f"[ERROR] Not found: {args.video}")
        return
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    roi_config = None
    
    if args.load_roi:
        selector = ROISelector()
        roi_config = selector.load_roi_from_file(args.load_roi)
    elif args.select_roi:
        cap = cv2.VideoCapture(0 if args.webcam else args.video)
        if not cap.isOpened():
            print("[ERROR] Cannot open")
            return
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("[ERROR] Cannot read frame")
            return
        
        selector = ROISelector()
        roi_config = selector.select_roi(frame)
        
        if roi_config:
            selector.save_roi_to_file(roi_config, args.save_roi)
        else:
            return
    
    if not roi_config:
        print("[ERROR] No ROI")
        return
    
    counter = FootfallCounter(
        model_path=args.model,
        roi_config=roi_config,
        confidence_threshold=args.confidence,
        webcam=args.webcam
    )
    
    counter.process_video(
        video_source=0 if args.webcam else args.video,
        output_path=args.output,
        show_realtime=not args.no_display
    )


if __name__ == '__main__':
    main()
