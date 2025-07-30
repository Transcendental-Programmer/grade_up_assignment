# models and training
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import time

np.random.seed(42)

class VoterPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feat_importance = {}
        
    def train_ensemble(self, X, y):
        print("training...")
        assert len(X) > 100, "need decent sample size"  # learned this the hard way
        start = time.time()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train_scaled, y_train)
        lr_cal = CalibratedClassifierCV(lr, method='sigmoid', cv=3)
        lr_cal.fit(X_train_scaled, y_train)
        self.models['lr'] = lr_cal
        
        # rf works better so giving it more weight in ensemble
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=20,
                                   min_samples_leaf=10, random_state=42)
        rf.fit(X_train, y_train)
        rf_cal = CalibratedClassifierCV(rf, method='sigmoid', cv=3)
        rf_cal.fit(X_train, y_train)
        self.models['rf'] = rf_cal
        
        lr_pred = lr_cal.predict(X_test_scaled)
        lr_prob = lr_cal.predict_proba(X_test_scaled)[:, 1]
        lr_auc = roc_auc_score(y_test, lr_prob)
        
        rf_pred = rf_cal.predict(X_test)
        rf_prob = rf_cal.predict_proba(X_test)[:, 1]
        rf_auc = roc_auc_score(y_test, rf_prob)
        
        print(f"lr: {lr_auc:.3f}")
        print(f"rf: {rf_auc:.3f}")
        
        self.feat_importance = dict(zip(X.columns, rf.feature_importances_))
        
        print(f"done in {time.time() - start:.1f}s")
        
    def predict_turnout(self, X):
        if X.shape[1] != 17:  # quick sanity check on features
            print(f"warning: expected 17 features, got {X.shape[1]}")
        
        try:
            X_scaled = self.scaler.transform(X)
        except:
            print("scaling failed - using raw features")
            X_scaled = X
        
        lr_prob = self.models['lr'].predict_proba(X_scaled)[:, 1]
        rf_prob = self.models['rf'].predict_proba(X)[:, 1]
        
        # 0.4/0.6 split based on our validation runs - rf consistently beat lr by ~3% auc
        ensemble_prob = 0.4 * lr_prob + 0.6 * rf_prob
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)
        
        return ensemble_pred, ensemble_prob
        
    def get_top_features(self, n=10):
        return sorted(self.feat_importance.items(), key=lambda x: x[1], reverse=True)[:n]
