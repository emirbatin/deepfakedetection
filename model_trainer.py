from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.pca = None
        self.feature_importance = None
        
    def preprocess_data(self, X):
        """Veri ön işleme"""
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        explained_variance_ratio = 0.95
        self.pca = PCA(n_components=explained_variance_ratio, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)
        
        print(f"PCA sonrası boyut: {X_pca.shape[1]} (Orijinal: {X.shape[1]})")
        
        return X_pca
        
    def train(self, X_train, y_train, X_valid=None, y_valid=None, plot_results=True):
        """Model eğitimi"""
        X_train_processed = self.preprocess_data(X_train)
        
        rf_params = {
            'n_estimators': [200, 300],
            'max_depth': [20, None],
            'min_samples_split': [2, 3],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        xgb_params = {
            'n_estimators': [200, 300],
            'max_depth': [3, 4],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'gamma': [0, 0.1]
        }
        
        gb_params = {
            'n_estimators': [200, 300],
            'max_depth': [3, 4],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 0.9],
            'min_samples_split': [2, 3]
        }
        
        print("Random Forest eğitimi başlıyor...")
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        rf_search = GridSearchCV(rf, rf_params, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc')
        rf_search.fit(X_train_processed, y_train)
        
        print("\nXGBoost eğitimi başlıyor...")
        xgb = XGBClassifier(random_state=42, n_jobs=-1)
        xgb_search = GridSearchCV(xgb, xgb_params, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc')
        xgb_search.fit(X_train_processed, y_train)
        
        print("\nGradient Boosting eğitimi başlıyor...")
        gb = GradientBoostingClassifier(random_state=42)
        gb_search = GridSearchCV(gb, gb_params, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc')
        gb_search.fit(X_train_processed, y_train)
        
        self.model = VotingClassifier(
            estimators=[
                ('rf', rf_search.best_estimator_),
                ('xgb', xgb_search.best_estimator_),
                ('gb', gb_search.best_estimator_)
            ],
            voting='soft',
            weights=[1, 1.2, 1]
        )
        
        self.model.fit(X_train_processed, y_train)
        
        if X_valid is not None and y_valid is not None:
            X_valid_processed = self.pca.transform(self.scaler.transform(X_valid))
            y_pred = self.model.predict(X_valid_processed)
            y_pred_proba = self.model.predict_proba(X_valid_processed)
            accuracy = accuracy_score(y_valid, y_pred)
            report = classification_report(y_valid, y_pred)
            conf_matrix = confusion_matrix(y_valid, y_pred)
            roc_auc = roc_auc_score(y_valid, y_pred_proba[:, 1])
            
            print("\n=== Model Performans Raporu ===")
            print(f"Doğruluk: {accuracy:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            print("\nSınıflandırma Raporu:")
            print(report)
            
            if plot_results:
                self._plot_results(conf_matrix, y_valid, y_pred_proba)
            
            rf_model = rf_search.best_estimator_
            feature_importance = pd.DataFrame({
                'feature': range(X_train_processed.shape[1]),
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            self.feature_importance = feature_importance
            
            return {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'report': report,
                'confusion_matrix': conf_matrix,
                'best_params': {
                    'rf': rf_search.best_params_,
                    'xgb': xgb_search.best_params_,
                    'gb': gb_search.best_params_
                }
            }
    
    def _plot_results(self, conf_matrix, y_valid, y_pred_proba):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Tahmin')
        plt.ylabel('Gerçek')
        
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_valid, y_pred_proba[:, 1])
        
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        
        plt.tight_layout()
        plt.show()
        
        if self.feature_importance is not None:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=self.feature_importance.head(20))
            plt.title('En Önemli 20 Özellik')
            plt.show()
    
    def save_model(self, model_path="deepfake_model.pkl"):
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş!")
            
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'pca': self.pca,
                'feature_importance': self.feature_importance
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model kaydedildi: {model_path}")
        except Exception as e:
            print(f"Model kaydetme hatası: {str(e)}")
    
    def load_model(self, model_path="deepfake_model.pkl"):
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.pca = model_data['pca']
                self.feature_importance = model_data.get('feature_importance', None)
            print(f"Model başarıyla yüklendi: {model_path}")
        except Exception as e:
            print(f"Model yükleme hatası: {str(e)}")

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş!")
            
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        predictions = self.model.predict(X_pca)
        probabilities = self.model.predict_proba(X_pca)
        
        return predictions, probabilities
