from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer
from predictor import DeepfakePredictor
from sklearn.metrics import accuracy_score
import pandas as pd

def main():
    try:
        # Özellik çıkarıcıyı başlat
        feature_extractor = FeatureExtractor()
        
        # Eğitim, doğrulama ve test veri kümelerini hazırla
        print("Eğitim verisi özellikleri çıkarılıyor...")
        X_train, y_train = feature_extractor.prepare_dataset("combined_dataset/train")
        
        print("Doğrulama verisi özellikleri çıkarılıyor...")
        X_valid, y_valid = feature_extractor.prepare_dataset("combined_dataset/valid")
        
        print("Test verisi özellikleri çıkarılıyor...")
        X_test, y_test = feature_extractor.prepare_dataset("combined_dataset/test")
        
        # Model eğitimini başlat
        print("Model eğitiliyor...")
        trainer = ModelTrainer()
        results = trainer.train(X_train, y_train, X_valid, y_valid)
        
        print("\n=== Model Sonuçları ===")
        print(f"Model doğruluğu: {results['accuracy']:.4f}")
        print(f"ROC AUC skoru: {results['roc_auc']:.4f}")
        print("\nSınıflandırma raporu:")
        print(results['report'])
        
        print("\nEn iyi parametreler:")
        for model_name, params in results['best_params'].items():
            print(f"\n{model_name} parametreleri:")
            for param, value in params.items():
                print(f"  {param}: {value}")
        
        # Modeli kaydet
        trainer.save_model()
        print("\nModel kaydedildi.")
        
        # Test veri kümesi ile tahmin yap ve doğruluğunu ölç
        y_pred, y_pred_proba = trainer.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest doğruluğu: {test_accuracy:.4f}")
        
        # Tahmin yapıcıyı başlat
        predictor = DeepfakePredictor(trainer, feature_extractor)
        
        # Örnek tahmin
        test_image = "mj.jpg"
        result = predictor.predict(test_image)
        
        if 'error' in result:
            print(f"\nTahmin hatası: {result['error']}")
        else:
            print(f"\nTest görüntüsü tahmini:")
            print(f"Sonuç: {result['prediction']}")
            print(f"Güven skoru: {result['confidence']:.4f}")
            
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()
