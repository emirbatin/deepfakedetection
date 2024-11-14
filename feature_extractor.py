import os
import cv2
import dlib
import numpy as np
from skimage.feature import graycomatrix, graycoprops

class FeatureExtractor:
    def __init__(self, predictor_path="models/shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        
    def calculate_distances(self, landmarks):
        """İşaret noktaları arası mesafeleri hesapla"""
        distances = []
        for i in range(68):
            for j in range(i+1, 68):
                x1, y1 = landmarks.part(i).x, landmarks.part(i).y
                x2, y2 = landmarks.part(j).x, landmarks.part(j).y
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                distances.append(distance)
        return distances
    
    def extract_glcm_features(self, gray):
        """Doku özellikleri çıkarma (GLCM)"""
        # Görüntüyü yeniden boyutlandır (hesaplama hızı için)
        resized = cv2.resize(gray, (128, 128))
        
        # GLCM parametreleri
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # GLCM matrisini hesapla
        glcm = graycomatrix(resized, distances=distances, angles=angles, 
                           levels=256, symmetric=True, normed=True)
        
        # GLCM özelliklerini çıkar
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        return [contrast, dissimilarity, homogeneity, energy, correlation]
    
    def extract_frequency_features(self, gray):
        """Frekans domain özellikleri"""
        # FFT uygula
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # İstatistiksel özellikler
        mean_freq = np.mean(magnitude_spectrum)
        std_freq = np.std(magnitude_spectrum)
        max_freq = np.max(magnitude_spectrum)
        median_freq = np.median(magnitude_spectrum)
        
        # Yüksek ve düşük frekans bölgeleri
        rows, cols = gray.shape
        center_row, center_col = rows//2, cols//2
        
        high_freq = magnitude_spectrum[center_row-10:center_row+10, 
                                     center_col-10:center_col+10].mean()
        low_freq = (magnitude_spectrum.mean() - high_freq)
        
        return [mean_freq, std_freq, max_freq, median_freq, high_freq, low_freq]
    
    def extract_color_features(self, image):
        """Renk tutarlılığı özellikleri"""
        # HSV renk uzayına dönüştür
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        features = []
        # Her kanal için istatistikler
        for channel in cv2.split(hsv):
            mean = np.mean(channel)
            std = np.std(channel)
            # Çarpıklık (Skewness)
            skew = np.mean((channel - mean)**3)
            skew = skew/((std**3) if std != 0 else 1)
            # Basıklık (Kurtosis)
            kurt = np.mean((channel - mean)**4)
            kurt = kurt/((std**4) if std != 0 else 1)
            # Entropi
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            hist = hist/hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            
            features.extend([mean, std, skew, kurt, entropy])
        
        return features
    
    def extract_quality_metrics(self, image, gray):
        """Görüntü kalite metrikleri"""
        # Bulanıklık değeri
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Gürültü tahmini
        noise = np.std(gray)
        
        # ELA (Error Level Analysis)
        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        compressed = cv2.imread(temp_path)
        ela = cv2.absdiff(image, compressed)
        ela_mean = np.mean(ela)
        os.remove(temp_path)
        
        # Kontrast
        contrast = np.std(gray)
        
        # Keskinlik
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sharpness = np.sqrt(sobelx**2 + sobely**2).mean()
        
        return [blur, noise, ela_mean, contrast, sharpness]
    
    def extract_features(self, image_path):
        """Tüm özellikleri çıkar"""
        # Görüntüyü oku
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # Gri tonlamalı görüntü
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None
            
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        features = []
        
        # 1. Yüz işaretleri
        landmarks_coords = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_coords.extend([x, y])
        features.extend(landmarks_coords)
        
        # 2. İşaretler arası mesafeler
        distances = self.calculate_distances(landmarks)
        features.extend(distances)
        
        # 3. GLCM doku özellikleri
        glcm_features = self.extract_glcm_features(gray)
        features.extend(glcm_features)
        
        # 4. Frekans özellikleri
        freq_features = self.extract_frequency_features(gray)
        features.extend(freq_features)
        
        # 5. Renk özellikleri
        color_features = self.extract_color_features(image)
        features.extend(color_features)
        
        # 6. Kalite metrikleri
        quality_features = self.extract_quality_metrics(image, gray)
        features.extend(quality_features)
        
        return features
        
    def prepare_dataset(self, dataset_path):
        """Veri setini hazırla"""
        data = []
        labels = []
        categories = ["real", "fake"]
        
        print("Özellikler çıkarılıyor...")
        total_images = sum([len(files) for _, _, files in os.walk(dataset_path)])
        processed = 0
        
        for category in categories:
            category_path = os.path.join(dataset_path, category)
            label = 1 if category == "fake" else 0
            
            for filename in os.listdir(category_path):
                image_path = os.path.join(category_path, filename)
                features = self.extract_features(image_path)
                
                if features is not None:
                    data.append(features)
                    labels.append(label)
                
                processed += 1
                if processed % 10 == 0:  # Her 10 görüntüde bir ilerleme göster
                    print(f"İşlenen görüntü: {processed}/{total_images}")
        
        return np.array(data), np.array(labels)