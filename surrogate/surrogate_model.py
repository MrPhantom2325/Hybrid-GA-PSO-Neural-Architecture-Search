
"""
surrogate/surrogate_model.py

Random Forest surrogate for predicting val_accuracy from chromosome features.
Keeps it intentionally simple — the paper contribution is the active-learning
loop, not the surrogate model itself.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score


class SurrogateModel:
    """
    Wraps a Random Forest regressor with a scaler.
    Predicts val_accuracy from the 13-dim chromosome feature vector.
    """

    def __init__(self, n_estimators=100, max_depth=6, random_state=42):
        self.rf = RandomForestRegressor(
            n_estimators = n_estimators,
            max_depth    = max_depth,
            random_state = random_state,
            n_jobs       = -1,
        )
        self.scaler  = StandardScaler()
        self.is_fit  = False
        self.r2_     = None
        self.mae_    = None
        self.n_train = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Fit surrogate. Returns train metrics.
        X: (N, 13) float32   y: (N,) float32
        """
        X_sc = self.scaler.fit_transform(X)
        self.rf.fit(X_sc, y)
        self.is_fit  = True
        self.n_train = len(y)

        y_pred       = self.rf.predict(X_sc)
        self.r2_     = r2_score(y, y_pred)
        self.mae_    = mean_absolute_error(y, y_pred)

        # Cross-validated R2 (only if enough samples)
        if len(y) >= 5:
            cv_r2 = cross_val_score(self.rf, X_sc, y, cv=min(5, len(y)), scoring="r2")
            self.cv_r2_ = float(cv_r2.mean())
        else:
            self.cv_r2_ = self.r2_

        return {
            "train_r2" : self.r2_,
            "train_mae": self.mae_,
            "cv_r2"    : self.cv_r2_,
            "n_samples": self.n_train,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict val_accuracy for a batch of chromosomes."""
        assert self.is_fit, "Call fit() first"
        X_sc = self.scaler.transform(X)
        return self.rf.predict(X_sc)

    def predict_with_std(self, X: np.ndarray):
        """
        Predict mean and std across trees.
        Std is a cheap uncertainty estimate — useful for upper-confidence-bound
        acquisition (optional extension for the paper).
        Returns: (mean_pred, std_pred) both shape (N,)
        """
        assert self.is_fit
        X_sc      = self.scaler.transform(X)
        tree_preds = np.array([tree.predict(X_sc) for tree in self.rf.estimators_])
        return tree_preds.mean(axis=0), tree_preds.std(axis=0)

    def feature_importances(self) -> np.ndarray:
        """Return RF feature importances (useful for paper ablation table)."""
        assert self.is_fit
        return self.rf.feature_importances_

    def is_reliable(self, min_r2=0.5) -> bool:
        """
        Returns True if surrogate CV R2 is above min_r2.
        Used to decide whether to trust predictions in the active-learning loop.
        With only 10–15 seed samples the surrogate will be noisy — that is expected.
        """
        return self.is_fit and (self.cv_r2_ >= min_r2)

    def __repr__(self):
        if self.is_fit:
            return (f"SurrogateModel(n={self.n_train}, "
                    f"train_R2={self.r2_:.3f}, cv_R2={self.cv_r2_:.3f}, "
                    f"MAE={self.mae_:.4f})")
        return "SurrogateModel(not fit)"
