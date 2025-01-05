import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import os
from datetime import datetime
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import mediapipe as mp

class DeepfakeDetectorApp:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Deepfake Detector")
        self.window.geometry("1000x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        try:
            with open("model.pkl", "rb") as f: 
                data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
        except Exception as e:
            messagebox.showerror("Hata", f"Model yükleme hatası: {e}")
            self.window.destroy()
            return
        
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        self.left_panel = ctk.CTkFrame(self.window, width=300)
        self.left_panel.pack(side="left", fill="y", padx=20, pady=20)
        
        self.title_label = ctk.CTkLabel(
            self.left_panel,
            text="Deepfake\nDetector",
            font=ctk.CTkFont(family="Helvetica", size=32, weight="bold"),
            text_color=("#1F6AA5", "#2D8AC3")
        )
        self.title_label.pack(pady=30)
        
        self.theme_switch = ctk.CTkSwitch(
            self.left_panel,
            text="Dark Mode",
            command=self.toggle_theme,
            font=ctk.CTkFont(size=14)
        )
        self.theme_switch.select()
        self.theme_switch.pack(pady=10)
        
        self.button_frame = ctk.CTkFrame(self.left_panel)
        self.button_frame.pack(pady=20, fill="x", padx=20)
        
        self.select_button = ctk.CTkButton(
            self.button_frame,
            text="Fotoğraf Seç",
            command=self.select_image,
            font=ctk.CTkFont(size=15),
            height=40
        )
        self.select_button.pack(pady=10, fill="x")
        
        self.detect_button = ctk.CTkButton(
            self.button_frame,
            text="Analiz Et",
            command=self.detect_deepfake,
            state="disabled",
            font=ctk.CTkFont(size=15),
            height=40
        )
        self.detect_button.pack(pady=10, fill="x")
        
        self.history_label = ctk.CTkLabel(
            self.left_panel,
            text="Analiz Geçmişi",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.history_label.pack(pady=(20, 10))
        
        self.history_text = ctk.CTkTextbox(
            self.left_panel,
            width=250,
            height=200,
            font=ctk.CTkFont(size=12)
        )
        self.history_text.pack(padx=10, pady=10)
        
        self.right_panel = ctk.CTkFrame(self.window)
        self.right_panel.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        
        self.image_frame = ctk.CTkFrame(self.right_panel)
        self.image_frame.pack(pady=20, expand=True)
        
        self.image_label = ctk.CTkLabel(self.image_frame, text="")
        self.image_label.pack(expand=True)
        
        self.result_label = ctk.CTkLabel(
            self.right_panel,
            text="Fotoğraf seçin ve analiz edin",
            font=ctk.CTkFont(size=18)
        )
        self.result_label.pack(pady=10)
        
        self.progress = ctk.CTkProgressBar(self.right_panel)
        self.progress.pack(pady=10, fill="x", padx=20)
        self.progress.set(0)
        
        self.selected_image_path = None
        
    def toggle_theme(self):
        if ctk.get_appearance_mode() == "Dark":
            ctk.set_appearance_mode("light")
        else:
            ctk.set_appearance_mode("dark")
    
    def extract_features(self, image_path):
        """MediaPipe ile yüz özelliklerini çıkarır ve eksik özellikleri doldurur."""
        image = cv2.imread(image_path)
        if image is None:
            return None
    
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None
    
        face_landmarks = results.multi_face_landmarks[0]
        features = []
    
        landmarks_3d = [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark]
        features.extend(np.array(landmarks_3d).flatten())
    
        expected_features = 1437
        if len(features) < expected_features:
            features.extend([0] * (expected_features - len(features)))
    
        return features
    

    def predict_image(self, image_path):
        """Tahmin yapar."""
        features = self.extract_features(image_path)
        if features is None:
            return {"error": "Yüz tespit edilemedi"}

        features = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        confidence = self.model.predict_proba(features_scaled)[0].max()

        return {
            "prediction": "real" if prediction == 0 else "fake",
            "confidence": confidence
        }
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")
            ]
        )
        
        if file_path:
            self.selected_image_path = file_path
            self.show_image(file_path)
            self.detect_button.configure(state="normal")
            self.result_label.configure(text="Fotoğraf seçildi. Analiz için hazır.")
            
    def show_image(self, image_path):
        pil_image = Image.open(image_path)
        width, height = pil_image.size
        aspect_ratio = width / height

        if width > height:
            new_width = 400
            new_height = int(400 / aspect_ratio)
        else:
            new_height = 400
            new_width = int(400 * aspect_ratio)

        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

        ctk_image = ctk.CTkImage(
            light_image=pil_image,
            dark_image=pil_image,
            size=(new_width, new_height)
        )

        self.image_label.configure(image=ctk_image)
        self.image_label.image = ctk_image
        
    def add_to_history(self, result):
        timestamp = datetime.now().strftime("%H:%M:%S")
        filename = os.path.basename(self.selected_image_path)
        prediction = result['prediction']
        confidence = result['confidence']
        
        history_entry = f"[{timestamp}] {filename[:20]}...\n"
        history_entry += f"Sonuç: {prediction.upper()}\n"
        history_entry += f"Güven: %{confidence*100:.2f}\n"
        history_entry += "-" * 30 + "\n"
        
        self.history_text.insert("1.0", history_entry)
        
    def detect_deepfake(self):
        if self.selected_image_path:
            self.progress.set(0.3)
            self.result_label.configure(
                text="Analiz ediliyor...",
                text_color=("gray60", "gray40")
            )
            self.window.update()
            
            try:
                result = self.predict_image(self.selected_image_path)
                self.progress.set(1.0)
                
                if 'error' in result:
                    self.result_label.configure(
                        text=f"Hata: {result['error']}",
                        text_color=("red", "#FF5555")
                    )
                else:
                    prediction = result['prediction']
                    confidence = result['confidence']
                    
                    if prediction == 'real':
                        color = ("green", "#55FF55")
                        text = "GERÇEK"
                    else:
                        color = ("red", "#FF5555")
                        text = "SAHTE"
                    
                    result_text = f"Sonuç: {text}\n"
                    result_text += f"Güven Skoru: %{confidence*100:.2f}"
                    
                    self.result_label.configure(
                        text=result_text,
                        text_color=color
                    )
                    
                    self.add_to_history(result)
            
            except Exception as e:
                self.result_label.configure(
                    text=f"Bir hata oluştu: {str(e)}",
                    text_color=("red", "#FF5555")
                )
                self.progress.set(0)
    
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = DeepfakeDetectorApp()
    app.run()
