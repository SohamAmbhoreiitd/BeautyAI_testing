import cv2
import mediapipe as mp
import itertools
import numpy as np
from scipy.interpolate import splprep, splev
from threading import Lock
import base64
import os
import torch
from torchvision import transforms
from PIL import Image
from torchvision import models
import torch.nn as nn

class MakeupApplication:
    def __init__(self):
        self._face_mesh = None
        self._face_mesh_lock = Lock()
        
        self.LIPS_INDEXES = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
            308, 409, 270, 269, 267, 0, 37, 39, 40, 185, 95
        ]

        # Skin Tone Detection Model Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @property
    def face_mesh(self):
        if self._face_mesh is None:
            with self._face_mesh_lock:
                if self._face_mesh is None:
                    self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5
                    )
        return self._face_mesh

    def apply_lipstick(self, image, landmarks, indexes, color, color_intensity=0.2):
        points = np.array([(int(landmarks[idx].x * image.shape[1]), int(landmarks[idx].y * image.shape[0])) for idx in indexes])
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        boundary_mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        
        colored_image = np.zeros_like(image)
        colored_image[:] = color
        
        lipstick_image = cv2.bitwise_and(colored_image, colored_image, mask=boundary_mask)
        lips_colored = cv2.addWeighted(image, 1, lipstick_image, color_intensity, 0)
        
        gradient_mask = cv2.GaussianBlur(boundary_mask, (15, 15), 0) / 255.0
        blurred = cv2.GaussianBlur(lips_colored, (7, 7), 3)
        
        lips_with_gradient = (blurred * gradient_mask[..., np.newaxis] + image * (1 - gradient_mask[..., np.newaxis])).astype(np.uint8)
        return np.where(boundary_mask[..., np.newaxis] != 0, lips_with_gradient, image)

    
    # In makeup_app.py, replace the apply_eyeliner function with this one.
    # In makeup_app.py, replace the apply_eyeliner function with this one.
    def apply_eyeliner(self, frame, color=(0, 0, 0), alpha=1, kajal=False, face_height=0):
        """
        Applies a realistic, tapered eyeliner with a subtle, elegant wing and correct direction. (FINAL TUNED VERSION)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            overlay = np.zeros_like(frame, dtype=np.uint8)

            lines = {
                'eyeliner': ([33, 246, 161, 160, 159, 158, 157, 173, 133], [263, 466, 388, 387, 386, 385, 384, 398, 362]),
                'kajal': ([33, 7, 163, 144, 145, 153, 154, 155, 133], [263, 249, 390, 373, 374, 380, 381, 382, 362])
            }
            line_type = 'kajal' if kajal else 'eyeliner'
            
            min_thickness = max(1, int(face_height / 250))
            max_thickness = max(2, int(face_height / 120))

            for face_landmarks in results.multi_face_landmarks:
                # Process each eye separately to ensure correct sorting direction
                
                # Eye on the left of the screen (Person's Right Eye)
                left_eye_indices = lines[line_type][0]
                left_points = np.array([[face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h] for idx in left_eye_indices], dtype=np.int32)
                left_points = left_points[left_points[:, 0].argsort()] # Sort ascending by x
                
                # Eye on the right of the screen (Person's Left Eye)
                right_eye_indices = lines[line_type][1]
                right_points = np.array([[face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h] for idx in right_eye_indices], dtype=np.int32)
                right_points = right_points[right_points[:, 0].argsort()[::-1]] # Sort descending by x
                
                for points in [left_points, right_points]:
                    # 1. Tapered Thickness
                    for i in range(len(points) - 1):
                        p1 = tuple(points[i])
                        p2 = tuple(points[i+1])
                        progress = (i + 1) / (len(points) - 1)
                        current_thickness = int(min_thickness + (max_thickness - min_thickness) * progress**2)
                        cv2.line(overlay, p1, p2, color, current_thickness, cv2.LINE_AA)

                    # 2. Winged Eyeliner Effect (More elegant and subtle)
                    if not kajal and len(points) > 1:
                        p_end = points[-1]
                        p_prev = points[-2]
                        
                        direction = (p_end[0] - p_prev[0], p_end[1] - p_prev[1])
                        norm = np.linalg.norm(direction)
                        if norm > 0:
                            # Reduced length and increased upward angle for a more elegant look
                            wing_length = face_height * 0.05 
                            wing_flick = wing_length * 0.6
                            wing_tip = (
                                int(p_end[0] + direction[0] * wing_length / norm),
                                int(p_end[1] + direction[1] * wing_length / norm - wing_flick)
                            )
                            wing_points = np.array([p_end, wing_tip, (p_end[0], p_end[1] + max_thickness)], dtype=np.int32)
                            cv2.fillPoly(overlay, [wing_points], color)

            # 3. Softer Edges
            blurred_overlay = cv2.GaussianBlur(overlay, (3, 3), 0)
            
            # Blend the final blurred overlay onto the original frame
            frame = cv2.addWeighted(blurred_overlay, alpha, frame, 1.0, 0)

        return frame
    def apply_eyeshadow(self, frame, left_eye_indices, right_eye_indices, color=(130, 50, 200), alpha=0.17):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            overlay = frame.copy()
            
            for face_landmarks in results.multi_face_landmarks:
                for eye_indices in [left_eye_indices, right_eye_indices]:
                    points = np.array([[int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)] for idx in eye_indices], np.int32)
                    cv2.fillPoly(overlay, [points], color)

            blurred_overlay = cv2.GaussianBlur(overlay, (35, 35), 0)
            frame = cv2.addWeighted(blurred_overlay, alpha, frame, 1 - alpha, 0)
        return frame
    
    def apply_blush(self, frame, left_cheek_indices, right_cheek_indices, color=(130, 119, 255), alpha=0.15):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            output_frame = frame.copy()
            overlay = frame.copy()

            for face_landmarks in results.multi_face_landmarks:
                for cheek_indices in [left_cheek_indices, right_cheek_indices]:
                    points = np.array([[int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)] for idx in cheek_indices], np.int32)
                    cv2.fillPoly(overlay, [points], color)

            blurred_overlay = cv2.GaussianBlur(overlay, (55, 55), 0)
            cv2.addWeighted(blurred_overlay, alpha, output_frame, 1 - alpha, 0, output_frame)
            return output_frame
        return frame

    def process_frame(self, frame, makeup_options=None):
        if makeup_options is None: makeup_options = {}
        
        processed_frame = frame.copy()
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            face_height_px = np.linalg.norm([
                (face_landmarks.landmark[152].x - face_landmarks.landmark[10].x) * w,
                (face_landmarks.landmark[152].y - face_landmarks.landmark[10].y) * h
            ])
            
            if makeup_options.get('lipstick', {}).get('enabled'):
                processed_frame = self.apply_lipstick(processed_frame, face_landmarks.landmark, self.LIPS_INDEXES, makeup_options['lipstick']['color'])
            if makeup_options.get('eyeliner', {}).get('enabled'):
                processed_frame = self.apply_eyeliner(processed_frame, makeup_options['eyeliner']['color'], kajal=False, face_height=face_height_px)
            if makeup_options.get('kajal', {}).get('enabled'):
                processed_frame = self.apply_eyeliner(processed_frame, makeup_options['kajal']['color'], kajal=True, face_height=face_height_px)
            if makeup_options.get('blush', {}).get('enabled'):
                indices = ([266, 330, 348, 449, 352, 411, 425], [36, 101, 119, 229, 123, 187, 205])
                processed_frame = self.apply_blush(processed_frame, indices[0], indices[1], makeup_options['blush']['color'])
            if makeup_options.get('eyeshadow', {}).get('enabled'):
                indices = ([33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
                           [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249])
                processed_frame = self.apply_eyeshadow(processed_frame, indices[0], indices[1], makeup_options['eyeshadow']['color'])
        return processed_frame

    def load_model(self):
        # This architecture must match the saved modelft.pth file.
        class_names = ['deep', 'light', 'mid-dark', 'mid-light']
        model_ft = models.mobilenet_v2(weights=None)
        num_ftrs = model_ft.classifier[1].in_features
        
        # This Sequential classifier is the correct structure for your model file.
        model_ft.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(num_ftrs, 50),
            nn.Dropout(0.5),
            nn.Linear(50, len(class_names))
        )
        
        model_path = os.path.join(os.path.dirname(__file__), "modelft.pth")
        model_ft.load_state_dict(torch.load(model_path, map_location=self.device))
        return model_ft.to(self.device)
    
    def extract_skin_color(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return {"success": False, "error": "No face detected"}
        
        pil_image = Image.fromarray(rgb_image)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)
        
        class_names = ['deep', 'light', 'mid-dark', 'mid-light']
        skin_type = class_names[predicted.item()]
        
        # Consolidate the model's output classes into the three you use
        if skin_type == "mid-dark":
            skin_type = "deep"
        elif skin_type == "mid-light":
            skin_type = "medium"
            
        return {"success": True, "skin_type": skin_type}

    def get_recommended_makeup_colors(self, skin_type):
        recommendations = {
            "deep": {
                "lipstick": [(213, 22, 87), (106, 8, 63), (63, 8, 31), (192, 96, 116), (55, 10, 18)],
                "blush": [(179, 57, 45), (243, 112, 33), (216, 27, 70), (173, 45, 96)],
                "eyeshadow": [(91,39,104), (237,115,38), (133,54,35), (181,87,49), (74,39,31), (24,86,49)],
                "eyeliner": [(29,28,26), (93,57,34)], "kajal":[(93,57,34),(46,91,188)]
            }, "light": { 
                "lipstick": [(217, 124, 111), (190, 104, 112), (247, 120, 111), (60, 5, 35)],
                "blush": [(242, 196, 167), (208, 132, 109), (251, 184, 180), (226, 157, 179)],
                "eyeshadow": [(205, 157, 210), (183, 159, 113), (255, 226, 209), (239, 109, 109)],
                "eyeliner": [(29,28,26), (93,57,34),(46,91,188)], "kajal": [(41,91,188),(93,57,34)]
            }, "medium": {
                "lipstick": [(165, 83, 52), (186, 102, 92), (160, 49, 32), (199, 104, 84)],
                "blush": [(233, 123, 103), (226, 87, 131), (143, 83, 79)],
                "eyeshadow": [(82,55,44), (143,106,89), (131,63,66), (177,105,85)],
                "eyeliner": [(29,28,26), (93,57,34)], "kajal": [(29,28,26),(93,57,34)]
            }
        }
        return recommendations.get(skin_type, {})