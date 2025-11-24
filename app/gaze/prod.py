import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from collections import deque, Counter
from pathlib import Path
import threading
from queue import Queue, Empty


# PART 1: MODEL ARCHITECTURES

# --- A. HopeNetLite (Head Pose) ---
try:
    from app.gaze.deep_head_pose_lite.hopenetlite_v2 import HopeNetLite
except ImportError:
    try:
        from deep_head_pose_lite.hopenetlite_v2 import HopeNetLite
    except ImportError:
        class HopeNetLite(nn.Module):
            def __init__(self):
                super(HopeNetLite, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(32)
                self.relu = nn.ReLU(inplace=True)
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                return x, x, x  # Dummy return for yaw, pitch, roll

# --- B. GazeAttentionNet (Gaze) ---
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

class GazeAttentionNet(nn.Module):
    def __init__(self, num_classes=2, use_depthwise=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        if use_depthwise:
            self.conv2 = nn.Sequential(DepthwiseSeparableConv(32, 64, stride=2), nn.ReLU(inplace=True))
            self.conv3 = nn.Sequential(DepthwiseSeparableConv(64, 128, stride=2), nn.ReLU(inplace=True))
            self.conv4 = nn.Sequential(DepthwiseSeparableConv(128, 128, stride=2), nn.ReLU(inplace=True))
        else:
            self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
            self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
            self.conv4 = nn.Sequential(nn.Conv2d(128, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# PART 2: HELPER CLASSES (INCL. WORKER)


class GazeInference:
    def __init__(self, model_path, backend='torchscript', device='cpu', img_size=64):
        self.model_path = str(model_path)
        self.device = device
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        try:
            print(f"Loading Gaze Model: {self.model_path}")
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            print(" - Success: Loaded as TorchScript.")
        except Exception as e_jit:
            print(f" - TorchScript load failed ({e_jit}), trying standard load...")
            self.model = GazeAttentionNet(num_classes=2, use_depthwise=True)
            self.model.to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            print(" - Success: Loaded as Standard PyTorch.")

    def predict(self, eye_crop):
        if len(eye_crop.shape) == 3 and eye_crop.shape[2] == 3:
            eye_crop = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
        
        img_tensor = self.transform(eye_crop).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            if isinstance(output, tuple): output = output[0]
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        label = int(np.argmax(probs))
        return {'label': label, 'probability': float(probs[label])}

# --- RESTORED GAZE WORKER FROM ORIGINAL CODE ---
class GazeWorker:
    def __init__(self, model_path, device='cpu'):
        self.input_queue = Queue(maxsize=3)
        self.output_queue = Queue(maxsize=3)
        self.running = False
        self.thread = None
        # Initialize detector inside worker to keep main thread free
        self.gaze_detector = GazeInference(model_path, device=device)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _loop(self):
        while self.running:
            try:
                eye_crop = self.input_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                res = self.gaze_detector.predict(eye_crop)
                try:
                    self.output_queue.put_nowait(res)
                except:
                    # Drop oldest if queue full (Original Logic)
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(res)
                    except: pass
            except Exception as e:
                print(f"Worker Error: {e}")

    def submit(self, eye_crop):
        try:
            self.input_queue.put_nowait(eye_crop)
        except: pass

    def get_result(self):
        try:
            return self.output_queue.get_nowait()
        except Empty:
            return None
    
    # Helper for synchronous calls (Backend mode)
    def predict_sync(self, eye_crop):
        return self.gaze_detector.predict(eye_crop)


class TrackerState:
    def __init__(self, vote_length=7, alpha=0.35):
        self.alpha = alpha
        self.yaw_prev = None
        self.pitch_prev = None
        self.roll_prev = None
        self.vote_buffer = deque(maxlen=vote_length)
        self.last_committed_state = "UNKNOWN"
        self.last_gaze_result = None # Store last async result
        
    def smooth_head_pose(self, yaw, pitch, roll):
        if self.yaw_prev is None:
            self.yaw_prev, self.pitch_prev, self.roll_prev = yaw, pitch, roll
            return yaw, pitch, roll
        yaw = self.alpha * yaw + (1 - self.alpha) * self.yaw_prev
        pitch = self.alpha * pitch + (1 - self.alpha) * self.pitch_prev
        roll = self.alpha * roll + (1 - self.alpha) * self.roll_prev
        self.yaw_prev, self.pitch_prev, self.roll_prev = yaw, pitch, roll
        return yaw, pitch, roll

    def update_gaze_vote(self, raw_state):
        self.vote_buffer.append(raw_state)
        if not self.vote_buffer: return self.last_committed_state
        candidate = Counter(self.vote_buffer).most_common(1)[0][0]
        count = sum(1 for v in self.vote_buffer if v == candidate)
        if candidate != self.last_committed_state:
            if count >= 3: self.last_committed_state = candidate
        else:
            self.last_committed_state = candidate
        return self.last_committed_state


# PART 3: MAIN LOGIC ENGINE


class TrackingEngine:
    def __init__(self, gaze_model_path, head_model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Engine on {self.device}")
        
        # Load Head Model
        self.head_model = HopeNetLite().to(self.device)
        print(f"Loading Head Model: {head_model_path}")
        state = torch.load(head_model_path, map_location=self.device, weights_only=False)
        self.head_model.load_state_dict(state, strict=False)
        self.head_model.eval()
        
        # Initialize Gaze Worker (Threaded)
        self.gaze_worker = GazeWorker(gaze_model_path, device='cpu')
        self.gaze_worker.start()
        
        # Initialize OpenCV face detector (more stable than MediaPipe)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.idx_tensor = torch.arange(66, dtype=torch.float).to(self.device)
        self.head_preprocess = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.internal_frame_count = 0

    def process_frame(self, frame, tracker_state, synchronous=False):
        """
        Args:
            synchronous (bool): 
                If False (Default): Uses frame skipping + threading (Original Logic).
                If True: Runs gaze IMMEDIATELY (For Backend/Single Photo).
        """
        self.internal_frame_count += 1
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using OpenCV Haar Cascade
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        
        if len(faces) == 0:
            return None
        
        # Use the first (largest) face
        x1, y1, fw, fh = faces[0]
        x2, y2 = x1 + fw, y1 + fh
        
        pad_w, pad_h = int(0.12 * (x2 - x1)), int(0.12 * (y2 - y1))
        x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
        x2, y2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
        face_rgb = rgb[y1:y2, x1:x2]
        face_bgr = frame[y1:y2, x1:x2]
        
        if face_rgb.size == 0: return None

        # Head Pose
        face_resized = cv2.resize(face_rgb, (224, 224))
        tensor = self.head_preprocess(face_resized).unsqueeze(0).to(self.device)
        with torch.no_grad():
            yaw_l, pitch_l, roll_l = self.head_model(tensor)
            yaw = (torch.sum(F.softmax(yaw_l, dim=1) * self.idx_tensor, dim=1).item() * 3) - 99
            pitch = (torch.sum(F.softmax(pitch_l, dim=1) * self.idx_tensor, dim=1).item() * 3) - 99
            roll = (torch.sum(F.softmax(roll_l, dim=1) * self.idx_tensor, dim=1).item() * 3) - 99

        yaw, pitch, roll = tracker_state.smooth_head_pose(yaw, pitch, roll)

        # --- GAZE LOGIC (THREADED vs SYNC) ---
        eye_crop = self._extract_eyes(face_bgr)
        gaze_result = tracker_state.last_gaze_result # Start with old result
        
        if eye_crop is not None:
            if synchronous:
                # Backend Mode: Force Run
                gaze_result = self.gaze_worker.predict_sync(eye_crop)
                tracker_state.last_gaze_result = gaze_result
            else:
                # Video Mode: Threading + Skipping (Original Logic)
                # 1. Submit every 3rd frame
                if self.internal_frame_count % 3 == 0:
                    self.gaze_worker.submit(eye_crop.copy())
                
                # 2. Check for new result
                new_res = self.gaze_worker.get_result()
                if new_res:
                    gaze_result = new_res
                    tracker_state.last_gaze_result = new_res

        # Determine State
        gaze_on_screen = True
        confidence = 0.0
        
        if gaze_result:
            gaze_on_screen = (gaze_result['label'] == 0)
            confidence = gaze_result['probability']
        
        # Logic
        if confidence < 0.6: state = "UNCERTAIN"
        elif abs(yaw) > 28 or abs(pitch) > 22: state = "AWAY (head)"
        elif not gaze_on_screen: state = "AWAY (eyes)"
        else: state = "ON_SCREEN"
        
        # For backend/synchronous mode: return immediate state without voting
        if synchronous:
            final_state = state
        else:
            # For video mode: use voting system
            final_state = tracker_state.update_gaze_vote(state)

        return {
            "bbox": [x1, y1, x2, y2],
            "head_pose": {"yaw": yaw, "pitch": pitch, "roll": roll},
            "attention": {"state": final_state, "confidence": confidence},
            "debug_eye": eye_crop
        }

    def _apply_clahe_bgr(self, bgr_img):
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _extract_eyes(self, face_bgr):
        face_enhanced = self._apply_clahe_bgr(face_bgr)
        face_gray = cv2.cvtColor(face_enhanced, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(
            face_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 10)
        )

        h, w = face_bgr.shape[:2]

        if len(eyes) == 0:
            y0, y1 = int(h * 0.20), int(h * 0.55)
            x0, x1 = int(w * 0.10), int(w * 0.90)
            eye_crop = face_enhanced[y0:y1, x0:x1]
            if eye_crop.size == 0: return None
            return cv2.resize(eye_crop, (64, 64))

        if len(eyes) == 1:
            (ex, ey, ew, eh) = eyes[0]
            pad_w, pad_h = int(0.25 * ew), int(0.35 * eh)
            x0, y0 = max(0, ex - pad_w), max(0, ey - pad_h)
            x1, y1 = min(w, ex + ew + pad_w), min(h, ey + eh + pad_h)
            eye_crop = face_enhanced[y0:y1, x0:x1]
            
        else:
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            ex1, ey1, ew1, eh1 = eyes_sorted[0]
            ex2, ey2, ew2, eh2 = eyes_sorted[-1]
            x0 = max(0, min(ex1, ex2) - int(0.15 * ew1))
            y0 = max(0, min(ey1, ey2) - int(0.2 * eh1))
            x1 = min(w, max(ex1 + ew1, ex2 + ew2) + int(0.15 * ew2))
            y1 = min(h, max(ey1 + eh1, ey2 + eh2) + int(0.2 * max(eh1, eh2)))
            eye_crop = face_enhanced[y0:y1, x0:x1]

        if eye_crop.size == 0: return None
        return cv2.resize(eye_crop, (64, 64))
    
    def cleanup(self):
        if self.gaze_worker:
            self.gaze_worker.stop()


# PART 4: WRAPPER FUNCTION (FOR BACKEND)

def analyze_image(frame, engine, user_state=None):
    if user_state is None:
        user_state = TrackerState()
    # Force synchronous=True for backend requests
    return engine.process_frame(frame, user_state, synchronous=True)


# PART 5: USAGE EXAMPLE

if __name__ == "__main__":
    # --- CONFIG ---
    GAZE_MODEL = r"C:\Users\DESKTOP\Downloads\eyegaze production\models\gaze_ts.pt"
    HEAD_MODEL = r"C:\Users\DESKTOP\Downloads\eyegaze production\deep_head_pose_lite\model\hopenet_lite_6MB.pkl"

    print("--- Starting System ---")
    
    try:
        # Initialize Engine
        engine = TrackingEngine(GAZE_MODEL, HEAD_MODEL)
        state = TrackerState()
        
        cap = cv2.VideoCapture(0)
        print("Camera Started. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret: break

            # Run Backend Logic (synchronous=False uses the original Threading/Skipping logic)
            result = engine.process_frame(frame, state, synchronous=False)

            if result:
                x1, y1, x2, y2 = result['bbox']
                yaw = result['head_pose']['yaw']
                pitch = result['head_pose']['pitch']
                roll = result['head_pose']['roll']
                att_state = result['attention']['state']
                conf = result['attention']['confidence']
                
                colors = {"ON_SCREEN": (0,255,0), "AWAY (head)": (0,0,255), "AWAY (eyes)": (0,165,255)}
                c = colors.get(att_state, (200,200,200))

                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                
                # Separate Lines Visuals
                head_text = f"Head: Yaw:{yaw:.0f} Pitch:{pitch:.0f} Roll:{roll:.0f}"
                cv2.putText(frame, head_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                gaze_text = f"Attention: {att_state} ({conf:.2f})"
                cv2.putText(frame, gaze_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
                
                if result.get('debug_eye') is not None:
                    h, w = frame.shape[:2]
                    try:
                        frame[10:74, w-74:w-10] = result['debug_eye']
                        cv2.rectangle(frame, (w-76, 8), (w-8, 76), c, 2)
                    except: pass

            cv2.imshow("Backend Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        engine.cleanup() # Stop threads
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()