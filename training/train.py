#!/usr/bin/env python3
"""
train_model.py

Production-grade PyTorch Training Pipeline for the AI-Powered Voice Transaction Authentication Engine.
Optimized for cross-platform deployments (NVIDIA Thor, TensorRT, Triton, and Apple Silicon MPS).

Final Enhancements:
- Syntax fixes for benchmark console printing.
- Safe CPU-bound ONNX export with `onnx.checker` verification.
- Integrated `torch.compile()` for CUDA (PyTorch 2.x) with safe base-model extraction.
- Configurable CuDNN deterministic benchmarking for training speed optimization.
- Added data validation for infinite values (`np.isinf`).
- Reduced benchmarking overhead (200 passes) and safe MPS cache clearing.
- Comprehensive artifact exporting (classification_report.json, training_history.json).
- Architecture-aligned saving mechanics and hyperparameter pulling.
"""


import os
import time
import json
import random
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import asdict
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import sklearn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
try:
    from torch.amp import autocast, GradScaler
    AMP_MODERN = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_MODERN = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
def amp_autocast(device_type):
    if AMP_MODERN:
        return torch.amp.autocast(device_type=device_type)
    return torch.amp.autocast()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    classification_report,
    roc_auc_score
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "data_path": PROJECT_ROOT / "training" / "data" / "dataset.csv",
    "artifacts_dir": PROJECT_ROOT / "training" / "models",
    'random_seed': 42,
    'checkpoint_version': "1.0.0",
    
    # Environment Optimization
    'deterministic': True,  # Set to False to enable CuDNN benchmarking for maximum speed

    
    # Dataset schema
    'features': [
        'account_age_days', 'kyc_verified', 'phone_verified', 'email_verified', 'voice_enrolled',
        'speaker_similarity', 'liveness_score', 'audio_quality', 'spoof_probability',
        'speech_rate_similarity', 'pronunciation_similarity', 'command_familiarity', 'stress_score', 'hesitation_score',
        'vehicle_speed', 'engine_running', 'location_familiarity', 'time_familiarity', 'driver_present', 'seatbelt_fastened',
        'previous_trust_score', 'failed_attempts', 'successful_transactions', 'fraud_history',
        'transaction_amount', 'transaction_category', 'beneficiary_type', 'beneficiary_frequency', 'transaction_risk',
        'intent_type', 'llm_confidence'
    ],
    'ignore_cols': ['user_id', 'transaction_id', 'trust_score', 'risk_score', 'confidence'],
    'target_col': 'decision',
    
    # Target Mapping
    'decisions': {
        0: 'ALLOW',
        1: 'VOICE_CHALLENGE',
        2: 'VOICE_AND_OTP',
        3: 'REJECT'
    }
}
# ==========================================
# HARDWARE CONFIGURATION RUNTIME HELPERS
# ==========================================
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Ensure deterministic operations across CPU and GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

def configure_environment() -> Tuple[torch.device, Dict[str, Any]]:
    """Detects system hardware capabilities and configures execution environments."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        workers = min(8, os.cpu_count() or 1)
        runtime_specs = {"num_workers": workers, "pin_memory": True, "persistent_workers": True}
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        workers = min(8, os.cpu_count() or 1)
        runtime_specs = {"num_workers": workers, "pin_memory": False, "persistent_workers": False}
        torch.set_float32_matmul_precision("high")
        logger.info("Apple Silicon MPS optimization parameters initialized ('high' precision matmul active).")
    else:
        device = torch.device("cpu")
        workers = min(8, os.cpu_count() or 1)
        runtime_specs = {"num_workers": workers, "pin_memory": False, "persistent_workers": False}
        
    logger.info(f"Execution Target Device set to: {str(device).upper()}")
    return device, runtime_specs

# ==========================================
# DATA & PREPROCESSING
# ==========================================
class AuthenticationDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.X = torch.FloatTensor(features)
        self.y = torch.LongTensor(targets)
        
    def __len__(self) -> int:
        return len(self.y)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
    
from engines.authentication_network import (
    AuthenticationNetwork,
    ModelConfig,
)


class DataPipeline:
    def __init__(self, config: Dict[str, Any], runtime_specs: Dict[str, Any]):
        self.config = config
        self.runtime_specs = runtime_specs
        self.scaler = StandardScaler()
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.categorical_cols = []
        
    def load_and_validate(self) -> pd.DataFrame:
        if not self.config['data_path'].exists():
            raise FileNotFoundError(f"Dataset missing at {self.config['data_path']}")
            
        assert all(c not in self.config['features'] for c in self.config['ignore_cols']), "ignore_cols overlap with features!"
            
        logger.info(f"Loading dataset from {self.config['data_path']}...")
        df = pd.read_csv(self.config['data_path'])
        missing = set(self.config["features"]) - set(df.columns)

        if missing:

            raise ValueError(

                f"Dataset missing columns: {sorted(missing)}"

            )
        # Validation checks for missing and infinite values
        if df[self.config['features'] + [self.config['target_col']]].isna().any().any():
            raise ValueError("Dataset contains missing (NaN) values in required features or targets.")
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if np.isinf(df[numeric_cols].to_numpy()).any():
            raise ValueError("Dataset contains infinite values (inf / -inf).")
            
        dupes = df.duplicated().sum()
        if dupes > 0:
            logger.warning(f"Found {dupes} duplicate rows. Dropping duplicates...")
            df = df.drop_duplicates()
            
        return df

    def process_and_split(
    self,
    df: pd.DataFrame,
    model_config: ModelConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, Dict[str, int]]:
        self.categorical_cols = df[self.config['features']].select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info("Dataset Target Distribution Summary:")
        dist = df[self.config['target_col']].value_counts()
        for k, v in self.config['decisions'].items():
            logger.info(f"  {v.ljust(20)} ........ {dist.get(k, 0)}")

        df_train, df_temp = train_test_split(
            df, test_size=0.30, stratify=df[self.config['target_col']], random_state=self.config['random_seed']
        )
        df_val, df_test = train_test_split(
            df_temp, test_size=0.50, stratify=df_temp[self.config['target_col']], random_state=self.config['random_seed']
        )
        
        sizes = {'Train': len(df_train), 'Val': len(df_val), 'Test': len(df_test), 'Total': len(df)}
        
        X_train, y_train = df_train[self.config['features']].copy(), df_train[self.config['target_col']].values
        X_val, y_val = df_val[self.config['features']].copy(), df_val[self.config['target_col']].values
        X_test, y_test = df_test[self.config['features']].copy(), df_test[self.config['target_col']].values

        if self.categorical_cols:
            X_train[self.categorical_cols] = self.ordinal_encoder.fit_transform(X_train[self.categorical_cols])
            X_val[self.categorical_cols] = self.ordinal_encoder.transform(X_val[self.categorical_cols])
            X_test[self.categorical_cols] = self.ordinal_encoder.transform(X_test[self.categorical_cols])

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        loader_kwargs = {
            'batch_size': model_config.batch_size, 
            'num_workers': self.runtime_specs['num_workers'], 
            'pin_memory': self.runtime_specs['pin_memory'], 
            'persistent_workers': self.runtime_specs['persistent_workers']
        }
        
        train_loader = DataLoader(AuthenticationDataset(X_train_scaled, y_train), shuffle=True, drop_last=False, **loader_kwargs)
        val_loader = DataLoader(AuthenticationDataset(X_val_scaled, y_val), shuffle=False, **loader_kwargs)
        test_loader = DataLoader(AuthenticationDataset(X_test_scaled, y_test), shuffle=False, **loader_kwargs)
        
        weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        return train_loader, val_loader, test_loader, weights, sizes

    def save_artifacts(self, model_config):
        self.config['artifacts_dir'].mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, self.config['artifacts_dir'] / 'scaler.pkl')
        if self.categorical_cols:
            joblib.dump(self.ordinal_encoder, self.config['artifacts_dir'] / 'encoder.pkl')
            
        class_mapping = {str(k): str(v) for k, v in self.config['decisions'].items()}
        with open(self.config['artifacts_dir'] / 'class_mapping.json', 'w') as f:
            json.dump(class_mapping, f, indent=4)
        
        numeric_cols = [

            c

            for c in self.config["features"]

            if c not in self.categorical_cols

        ]

        feature_columns_info = {

            "features": self.config["features"],

            "categorical_cols": self.categorical_cols,

            "numeric_cols": numeric_cols,

        }
        with open(self.config['artifacts_dir'] / 'feature_columns.json', 'w') as f:
            json.dump(feature_columns_info, f, indent=4)
        
        model_info = {
            'batch_size':model_config.batch_size,
            'max_epochs':model_config.max_epochs,
            "input_dim": model_config.num_features,
            "num_features": model_config.num_features,
            "num_classes": model_config.num_decision_classes,
            "feature_order": self.config["features"],
            "categorical_cols": self.categorical_cols,

            "embedding_dim": model_config.embedding_dim,
            "projection_dim": model_config.proj_dim,
            "decision_classes": model_config.num_decision_classes,

            "dropout": model_config.dropout,
            "has_encoder": len(self.categorical_cols) > 0,
            "environment_metadata": {
                "sklearn_version": sklearn.__version__,
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
            },
        }
        with open(self.config['artifacts_dir'] / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=4)

class DecisionWrapper(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(x).decision_logits
    
# ==========================================
# TRAINER & BENCHMARK SUITE
# ==========================================
class ModelTrainer:
    def __init__(self, model: nn.Module, device: torch.device, class_weights: np.ndarray, config: Dict[str, Any], total_params: int):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.total_params = total_params
        
        # Safely extract hyper-params from base model regardless of compilation wrapping
        base_model = getattr(self.model, "_orig_mod", self.model)
        
        weight_tensor = torch.FloatTensor(class_weights).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=base_model.config.lr, 
            weight_decay=base_model.config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=base_model.config.t_max
        )
        
        self.grad_clip_norm = base_model.config.grad_clip_norm
        self.early_stopping_patience = base_model.config.early_stopping_patience
        
        self.use_amp = self.device.type == 'cuda'
        if self.use_amp:
            if AMP_MODERN:
                self.scaler = GradScaler(device="cuda", enabled=True)
            else:
                self.scaler = GradScaler(enabled=True)
        else:
            self.scaler = None

        self.best_macro_f1 = 0.0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = []
        self.start_training_time = 0
        self.total_time_min = 0.0

    def _sync_hardware(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elif self.device.type == 'mps':
            try:
                torch.mps.synchronize()
            except AttributeError:
                pass
    def _get_serializable_config(self):

        cfg = self.config.copy()

        cfg["data_path"] = str(cfg["data_path"])
        cfg["artifacts_dir"] = str(cfg["artifacts_dir"])

        return cfg

    def _compute_metrics(self, y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
        y_pred = y_probs.argmax(axis=1)
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        _, _, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_probs, multi_class="ovr")
        except ValueError:
            auc = float("nan")
        
        return {
            'Accuracy': float(acc), 'Precision': float(prec), 'Recall': float(rec), 
            'Macro F1': float(f1_macro), 'Weighted F1': float(f1_weighted), 'ROC AUC': float(auc)
        }

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0
        all_probs, all_targets = [], []
        
        for X, y in tqdm(loader, desc="Training", leave=False):
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.use_amp:
                with amp_autocast(self.device.type):
                    prediction = self.model(X)
                    loss = self.criterion(
                        prediction.decision_logits,
                        y
                    )
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                prediction = self.model(X)
                loss = self.criterion(
                    prediction.decision_logits,
                    y
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
                
            total_loss += loss.item()
            all_probs.extend(
                torch.softmax(
                    prediction.decision_logits,
                    dim=1
                )
                .detach()
                .cpu()
                .numpy()
            )
            all_targets.extend(y.cpu().numpy())
            
        metrics = self._compute_metrics(np.array(all_targets), np.array(all_probs))
        return total_loss / len(loader), metrics

    @torch.inference_mode()
    def _validate_epoch(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0
        all_probs, all_targets = [], []
        
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            
            if self.use_amp:
                with amp_autocast(self.device.type):
                    prediction = self.model(X)
                    loss = self.criterion(
                        prediction.decision_logits,
                        y
                    )
            else:
                prediction = self.model(X)
                loss = self.criterion(
                        prediction.decision_logits,
                        y
                    )
                
            total_loss += loss.item()
            all_probs.extend(torch.softmax(prediction.decision_logits, dim=1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
        metrics = self._compute_metrics(np.array(all_targets), np.array(all_probs))
        return total_loss / len(loader), metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            
        self.start_training_time = time.time()
        logger.info(f"Training Pipeline Started. Target Optimization Version: {self.config['checkpoint_version']}")
        base_model = getattr(self.model, "_orig_mod", self.model)
        self.max_epochs = base_model.config.max_epochs
        for epoch in range(1, self.max_epochs + 1):
            epoch_start_time = time.time()
            
            train_loss, train_metrics = self._train_epoch(train_loader)
            val_loss, val_metrics = self._validate_epoch(val_loader)
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            val_macro_f1 = val_metrics['Macro F1']
            
            if self.device.type == 'cuda':
                gpu_mem = float(torch.cuda.max_memory_allocated() / (1024**2))
                mem_log_str = f"{gpu_mem:.1f} MB (CUDA)"
            elif self.device.type == 'mps':
                gpu_mem = None
                mem_log_str = "N/A (MPS)"
            else:
                gpu_mem = None
                mem_log_str = "N/A (CPU)"
            
            logger.info(
                f"Epoch {epoch:03d}/{self.max_epochs} | LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_macro_f1:.4f} | Memory: {mem_log_str}"
            )
            
            self.history.append({
                'epoch': epoch, 'lr': current_lr, 'epoch_time_sec': epoch_time, 'gpu_memory_mb': gpu_mem,
                'train_loss': train_loss, 'val_loss': val_loss,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })
            
            improved_f1 = val_macro_f1 > self.best_macro_f1
            similar_f1_better_loss = abs(val_macro_f1 - self.best_macro_f1) < 1e-4 and val_loss < self.best_val_loss
            
            if improved_f1 or similar_f1_better_loss:
                self.best_macro_f1 = max(self.best_macro_f1, val_macro_f1)
                self.best_val_loss = min(self.best_val_loss, val_loss)
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Fetch original model if wrapped by torch.compile()
                base_model = getattr(self.model, "_orig_mod", self.model)
                
                torch.save({
                    "checkpoint_version": self.config['checkpoint_version'],
                    "epoch": epoch,
                    "model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "best_macro_f1": self.best_macro_f1,
                    "best_val_loss": self.best_val_loss,
                    "feature_order": self.config['features'],
                    "config": asdict(base_model.config)
                }, self.config['artifacts_dir'] / 'best_model.pth')
                logger.info(f"  --> Checkpoint Updated (Macro F1: {self.best_macro_f1:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    logger.warning(f"Early stopping triggered at epoch {epoch}.")
                    break
                    
        self.total_time_min = (time.time() - self.start_training_time) / 60
        
        # Load best weights (extracted to base_model for safe cross-platform saving)
        checkpoint = torch.load(self.config['artifacts_dir'] / 'best_model.pth', map_location=self.device, weights_only=False)
        base_model = getattr(self.model, "_orig_mod", self.model)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        torch.save(base_model.state_dict(), self.config['artifacts_dir'] / 'authentication_model.pth')
        
        # Export structural compilation engines using the safe uncompiled base model
        base_model.eval()
        
        # 1. Structural compilation using TorchScript JIT
        try:
            wrapper = DecisionWrapper(base_model).to(self.device)
            scripted_model = torch.jit.script(wrapper)
            scripted_model.save(self.config['artifacts_dir'] / 'authentication_model.pt')
            logger.info("  --> Structural TorchScript Engine Exported (.pt)")
        except Exception as e:
            logger.error(f"TorchScript scripting failed ({e}), falling back to trace mode stabilization...")
            example_fallback = torch.randn(1, base_model.config.num_features).to(self.device)
            wrapper = DecisionWrapper(base_model).to(self.device)
            torch.jit.trace(
                wrapper,
                example_fallback
            ).save(self.config['artifacts_dir'] / 'authentication_model.pt')
            logger.info("  --> Traced TorchScript Engine Exported (.pt)")

        # 2. Universal ONNX Export Engine for TensorRT / Triton (Run on CPU for ultimate reliability)
        import copy

        model_cpu = copy.deepcopy(base_model).cpu()
        example_onnx = torch.randn(1, base_model.config.num_features)
        onnx_path = self.config['artifacts_dir'] / 'authentication_model.onnx'
        
        wrapper = DecisionWrapper(model_cpu)


        wrapper.eval()
        torch.onnx.export(
            wrapper,
            example_onnx,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        logger.info("  --> TensorRT Optimization Target Engine Exported (.onnx)")
        
        # Restore model back to training device context
        base_model.to(self.device)

        # 3. Verify ONNX Output
        try:
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            logger.info("  --> ONNX Model successfully verified with onnx.checker.")
        except ImportError:
            logger.warning("  --> 'onnx' package not found. Skipping ONNX checker validation.")
        except Exception as e:
            logger.error(f"  --> ONNX verification failed: {e}")

        # Clear hardware working allocations safely
        if self.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
            
        pd.DataFrame(self.history).to_csv(self.config['artifacts_dir'] / 'training_history.csv', index=False)
        with open(self.config['artifacts_dir'] / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=4)

    @torch.inference_mode()
    def run_inference_benchmark(self, batch_size: int, num_passes: int = 200) -> Dict[str, float]:
        self.model.eval()
        base_model = getattr(self.model, "_orig_mod", self.model)
        example = torch.randn(batch_size, base_model.config.num_features).to(self.device)
        
        for _ in range(50):
            _ = self.model(example).decision_logits
            
        latencies = []
        for _ in range(num_passes):
            self._sync_hardware() 
            start = time.perf_counter()
            _ = self.model(example).decision_logits
            self._sync_hardware() 
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
            
        latencies = np.array(latencies)
        avg_latency = np.mean(latencies)
        
        return {
            "Average Latency (ms)": float(avg_latency),
            "P50 Latency (ms)": float(np.percentile(latencies, 50)),
            "P95 Latency (ms)": float(np.percentile(latencies, 95)),
            "FPS (Throughput)": float((1000.0 / avg_latency) * batch_size if avg_latency > 0 else 0.0)
        }

    @torch.inference_mode()
    def evaluate(self, test_loader: DataLoader):
        logger.info("Running Final Evaluation Matrix on Validation Sets...")
        self.model.eval()
        all_probs, all_targets = [], []
        
        for X, y in test_loader:
            X, y = X.to(self.device), y.to(self.device)
            prediction = self.model(X)

            all_probs.extend(
                torch.softmax(
                    prediction.decision_logits,
                    dim=1
                )
                .cpu()
                .numpy()
            )
            all_targets.extend(y.cpu().numpy())
            
        y_true, y_probs = np.array(all_targets), np.array(all_probs)
        y_pred = y_probs.argmax(axis=1)
        
        metrics = self._compute_metrics(y_true, y_probs)
        cm = confusion_matrix(y_true, y_pred)
        target_names = [self.config['decisions'][i] for i in range(len(self.config['decisions']))]
        
        cr_str = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
        cr_dict = classification_report(y_true, y_pred, target_names=target_names, zero_division=0, output_dict=True)
        
        # Output Generation
        np.save(self.config['artifacts_dir'] / 'confusion_matrix.npy', cm)
        pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(self.config['artifacts_dir'] / 'confusion_matrix.csv', index=True)
        
        with open(self.config['artifacts_dir'] / 'classification_report.json', 'w') as f:
            json.dump(cr_dict, f, indent=4)
        
        benchmark_batch_sizes = [1, 8, 16, 32, 64]
        benchmark_results = {}
        logger.info("Executing production hardware scaling grid benchmark tests...")
        for bs in benchmark_batch_sizes:
            benchmark_results[f"batch_size_{bs}"] = self.run_inference_benchmark(batch_size=bs, num_passes=200)
            
        summary_payload = {
            "checkpoint_version": self.config['checkpoint_version'],
            "best_epoch": int(self.best_epoch),
            "best_macro_f1": float(self.best_macro_f1),
            "total_epochs_run": int(len(self.history)),
            "training_time_minutes": float(self.total_time_min),
            "device_executed": str(self.device.type).upper(),
            "total_model_parameters": int(self.total_params),
            "scaling_inference_benchmarks": benchmark_results
        }
        
        with open(self.config['artifacts_dir'] / 'training_summary.json', 'w') as f:
            json.dump(summary_payload, f, indent=4)
            
        with open(self.config['artifacts_dir'] / 'training_metrics.json', 'w') as f:
            json.dump({'test_metrics': metrics}, f, indent=4)
            
        # UI Terminal Display Report Output
        print("\n" + "="*60)
        print("PRODUCTION PIPELINE TRAINING RESULTS SUMMARY")
        print("="*60)
        print(f"Device Executed:         {summary_payload['device_executed']}")
        print(f"Total Pipeline Runtime:  {summary_payload['training_time_minutes']:.2f} Minutes")
        print(f"Best Validation Epoch:   {summary_payload['best_epoch']} (F1: {summary_payload['best_macro_f1']:.4f})")
        print(f"Target Accuracy Rate:    {metrics['Accuracy']:.4f} | ROC-AUC: {metrics['ROC AUC']:.4f}")
        
        print("\n" + "="*60)
        print("SCALING LATENCY HARDWARE PROFILE MATRIX")
        print("="*60)
        print(f"{'Batch Size'.ljust(12)} | {'Avg Latency'.ljust(15)} | {'P95 Latency'.ljust(15)} | {'Throughput (FPS)'}")
        print("-" * 60)
        for bs in benchmark_batch_sizes:
            data = benchmark_results[f"batch_size_{bs}"]
            avg_ms = f"{data['Average Latency (ms)']:.3f} ms"
            p95_ms = f"{data['P95 Latency (ms)']:.3f} ms"
            print(f"{str(bs).ljust(12)} | {avg_ms.ljust(15)} | {p95_ms.ljust(15)} | {data['FPS (Throughput)']:.1f}")

        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(cr_str)
        print("="*60 + "\n")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    set_seed(CONFIG['random_seed'], deterministic=CONFIG['deterministic'])
    device, runtime_specs = configure_environment()
    
    pipeline = DataPipeline(CONFIG, runtime_specs)
    df = pipeline.load_and_validate()

    CONFIG_PATH = PROJECT_ROOT / "config" / "auth" / "config.yaml"

    model_config = ModelConfig.from_yaml(CONFIG_PATH)

    train_loader, val_loader, test_loader, class_weights, sizes = (
        pipeline.process_and_split(df, model_config)
    )

    logger.info(
        f"Dataset Split -> "
        f"Train: {sizes['Train']} | "
        f"Val: {sizes['Val']} | "
        f"Test: {sizes['Test']} | "
        f"Total: {sizes['Total']}"
    )

    model = AuthenticationNetwork(model_config)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Save artifacts properly after model config initialization
    pipeline.save_artifacts(model_config)
    
    # Enable torch.compile() for PyTorch 2.x CUDA acceleration
    if hasattr(torch, "compile") and device.type == "cuda":
        logger.info("Applying torch.compile() for CUDA graph optimization...")
        model = torch.compile(model)
    
    trainer = ModelTrainer(model, device, class_weights, CONFIG, total_params)
    trainer.train(train_loader, val_loader)
    trainer.evaluate(test_loader)

if __name__ == "__main__":
    main()