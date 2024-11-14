from feature_extractor import FeatureExtractor

class DeepfakePredictor:
    def __init__(self, model_trainer, feature_extractor):
        self.model_trainer = model_trainer
        self.feature_extractor = feature_extractor
        
    def predict(self, image_path):
        """Görüntü için tahmin yap"""
        # Özellikleri çıkar
        features = self.feature_extractor.extract_features(image_path)
        if features is None:
            return {
                'error': 'Görüntüde yüz bulunamadı veya görüntü okunamadı.'
            }
        
        try:
            # Özellikleri ölçeklendir ve PCA uygula
            features_scaled = self.model_trainer.scaler.transform([features])
            features_pca = self.model_trainer.pca.transform(features_scaled)
            
            # Tahmin yap
            prediction = self.model_trainer.model.predict(features_pca)
            probability = self.model_trainer.model.predict_proba(features_pca)[0][1]
            
            return {
                'prediction': 'fake' if prediction[0] == 1 else 'real',
                'confidence': probability
            }
        except Exception as e:
            return {
                'error': f'Tahmin sırasında hata oluştu: {str(e)}'
            }