import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import os
from predictor import DeepfakePredictor
from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer
from datetime import datetime

class DeepfakeDetectorApp:
    def __init__(self):
        # Ana pencere ayarları
        self.window = ctk.CTk()
        self.window.title("Deepfake Detector")
        self.window.geometry("1000x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Model yükleme
        self.feature_extractor = FeatureExtractor()
        self.trainer = ModelTrainer()
        self.trainer.load_model()
        self.predictor = DeepfakePredictor(self.trainer, self.feature_extractor)
        
        # Sol panel (Kontroller)
        self.left_panel = ctk.CTkFrame(self.window, width=300)
        self.left_panel.pack(side="left", fill="y", padx=20, pady=20)
        
        # Logo ve başlık
        self.title_label = ctk.CTkLabel(
            self.left_panel,
            text="Deepfake\nDetector",
            font=ctk.CTkFont(family="Helvetica", size=32, weight="bold"),
            text_color=("#1F6AA5", "#2D8AC3")
        )
        self.title_label.pack(pady=30)
        
        # Tema değiştirme
        self.theme_switch = ctk.CTkSwitch(
            self.left_panel,
            text="Dark Mode",
            command=self.toggle_theme,
            font=ctk.CTkFont(size=14)
        )
        self.theme_switch.select()  # Varsayılan dark mode
        self.theme_switch.pack(pady=10)
        
        # Butonlar için frame
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
        
        # Analiz geçmişi
        self.history_label = ctk.CTkLabel(
            self.left_panel,
            text="Analiz Geçmişi",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.history_label.pack(pady=(20,10))
        
        self.history_text = ctk.CTkTextbox(
            self.left_panel,
            width=250,
            height=200,
            font=ctk.CTkFont(size=12)
        )
        self.history_text.pack(padx=10, pady=10)
        
        # Sağ panel (Görüntü ve sonuçlar)
        self.right_panel = ctk.CTkFrame(self.window)
        self.right_panel.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        
        # Görüntü gösterme alanı
        self.image_frame = ctk.CTkFrame(self.right_panel)
        self.image_frame.pack(pady=20, expand=True)
        
        self.image_label = ctk.CTkLabel(self.image_frame, text="")
        self.image_label.pack(expand=True)
        
        # Sonuç gösterme alanı
        self.result_frame = ctk.CTkFrame(self.right_panel)
        self.result_frame.pack(pady=20, fill="x", padx=20)
        
        self.result_label = ctk.CTkLabel(
            self.result_frame,
            text="Fotoğraf seçin ve analiz edin",
            font=ctk.CTkFont(size=18)
        )
        self.result_label.pack(pady=10)
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(self.right_panel)
        self.progress.pack(pady=10, fill="x", padx=20)
        self.progress.set(0)
        
        self.selected_image_path = None
        
    def toggle_theme(self):
        if ctk.get_appearance_mode() == "Dark":
            ctk.set_appearance_mode("light")
        else:
            ctk.set_appearance_mode("dark")
    
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
        # Görüntüyü yükle
        pil_image = Image.open(image_path)
        
        # En boy oranını koru ve maksimum 400x400 yap
        width, height = pil_image.size
        aspect_ratio = width / height
        
        if width > height:
            new_width = 400
            new_height = int(400 / aspect_ratio)
        else:
            new_height = 400
            new_width = int(400 * aspect_ratio)
        
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # CTkImage olarak dönüştür
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
                result = self.predictor.predict(self.selected_image_path)
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
                    
                    # Geçmişe ekle
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