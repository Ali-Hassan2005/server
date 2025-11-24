from fastapi import UploadFile
from app.services.face_service import count_faces_in_frame, is_match
from app.services.Object_detection_service import detect
from app.services.gaze_attention_service import gaze_service
from typing import Optional


class ProctoringController:

    async def detect_cheating(self, image: UploadFile, reference_image: Optional[UploadFile] = None):

        try:
            # Count faces in the frame
            face_count = count_faces_in_frame(image)
            
            # Reset file pointer to read again for face matching
            await image.seek(0)
            
            # Check face match if reference image provided
            face_match_result = None
            if reference_image:
                face_match_result = is_match(reference_image, image)
                await image.seek(0)
            
            # Detect objects (phones, headphones, etc.)
            object_detection_result = detect(image)
            
            # Reset file pointer for gaze analysis
            await image.seek(0)
            
            # Analyze gaze and attention
            gaze_result = await gaze_service.analyze_attention(image)
            
            # Reset file pointer for potential future use
            await image.seek(0)
            
            # Determine if cheating detected
 
            reasons = []
            
            if face_count == 0:
                cheating_detected = True
                reasons.append("No face detected in frame")
            elif face_count > 1:
 
                reasons.append(f"Multiple faces detected: {face_count}")
            
            # Check face match result
            if face_match_result:
                if face_match_result.get("error"):
                    reasons.append(f"Face verification error: {face_match_result['error']}")
                elif not face_match_result.get("match"):
 
                    reasons.append("Face does not match reference image")
            
            if "error" not in object_detection_result and object_detection_result.get("count", 0) > 0:
               
                detected_objects = [d["class_name"] for d in object_detection_result.get("detections", [])]
                reasons.append(f"Prohibited objects detected: {', '.join(detected_objects)}")
            
            # Check gaze/attention result
            if gaze_result.get("error"):
                reasons.append(f"Gaze analysis error: {gaze_result['error']}")
            elif gaze_result.get("is_distracted"):
                reasons.append(f"User distracted: {gaze_result.get('attention_state', 'Unknown')}")
            
            return {
                "face_count": face_count,
                "face_match": face_match_result['match'] if face_match_result else None,
                "object_detection": object_detection_result,
                "gaze_attention": gaze_result
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "cheating_detected": None
            }
         
    

proctoring_controller = ProctoringController()
