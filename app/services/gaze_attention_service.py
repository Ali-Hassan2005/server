from fastapi import UploadFile
import cv2
import numpy as np
from pathlib import Path
from app.gaze.prod import TrackingEngine, TrackerState, analyze_image


class GazeAttentionService:
    """
    Service for detecting user attention through gaze and head pose tracking.
    """
    
    def __init__(self):
        """Initialize service (engine loaded lazily on first use)."""
        self._engine = None
        self._initialized = False
    
    def _get_engine(self):
        """Lazy load the tracking engine."""
        if not self._initialized:
            base_path = Path(__file__).parent.parent
            gaze_model_path = base_path / "ai_models" / "gaze_ts.pt"
            head_model_path = base_path / "gaze" / "deep_head_pose_lite" / "model" / "hopenet_lite_6MB.pkl"
            
            # Initialize the tracking engine
            self._engine = TrackingEngine(
                gaze_model_path=str(gaze_model_path),
                head_model_path=str(head_model_path)
            )
            self._initialized = True
        return self._engine
    
    async def analyze_attention(self, image: UploadFile) -> dict:
        """
        Analyze user attention from an uploaded image.
        
        Args:
            image: FastAPI UploadFile containing the image to analyze
            
        Returns:
            dict: Analysis results containing:
                - attention_state: "ON_SCREEN", "AWAY (head)", "AWAY (eyes)", or "UNCERTAIN"
                - confidence: Probability score (0.0-1.0)
                - head_pose: yaw, pitch, roll angles
                - is_distracted: Boolean indicating if user is not paying attention
                - error: Error message if processing failed
        """
        try:
            # Read image from upload
            contents = await image.read()
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {
                    "error": "Failed to decode image",
                    "attention_state": "UNKNOWN",
                    "is_distracted": None
                }
            
            # Create tracker state for this single frame analysis
            tracker_state = TrackerState()
            
            # Get engine (lazy load on first use)
            engine = self._get_engine()
            
            # Process frame synchronously (backend mode)
            result = analyze_image(frame, engine, tracker_state)
            
            if result is None:
                return {
                    "error": "No face detected in frame",
                    "attention_state": "UNKNOWN",
                    "is_distracted": True
                }
            
            # Extract results
            attention_state = result['attention']['state']
            confidence = result['attention']['confidence']
            head_pose = result['head_pose']
            
            # Determine if user is distracted
            is_distracted = attention_state in ["AWAY (head)", "AWAY (eyes)"]
            
            return {
                "attention_state": attention_state,
                "confidence": confidence,
                "head_pose": {
                    "yaw": round(head_pose['yaw'], 2),
                    "pitch": round(head_pose['pitch'], 2),
                    "roll": round(head_pose['roll'], 2)
                },
                "is_distracted": is_distracted,
                "description": self._get_state_description(attention_state)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "attention_state": "UNKNOWN",
                "is_distracted": None
            }
    
    def _get_state_description(self, state: str) -> str:
        """Get human-readable description of attention state."""
        descriptions = {
            "ON_SCREEN": "User is focused on screen",
            "AWAY (head)": "User's head is turned away from screen",
            "AWAY (eyes)": "User is looking away from screen",
            "UNCERTAIN": "Unable to determine attention with confidence"
        }
        return descriptions.get(state, "Unknown state")


# Singleton instance
gaze_service = GazeAttentionService()
