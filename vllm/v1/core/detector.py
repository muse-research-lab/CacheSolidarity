# import json
# from collections import defaultdict
# from datetime import datetime
# import os

# import json
# from collections import defaultdict
# from datetime import datetime
# import os
# from typing import List, Dict, Any, Optional


# class BlockEventLogger:
#     def __init__(self, output_dir: Optional[str] = None):
#         """
#         Simple JSONL logger for block events. Events are stored on-disk and
#         in-memory keyed by request_id for quick inspection.

#         :param output_dir: directory where log file will be written
#         """
#         if output_dir is None:
#             output_dir = "./home/panagiotis.pennas/vllm.v1/vllm/panos_experiments/workloads/datasets/diagram/"

#         os.makedirs(output_dir, exist_ok=True)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.filepath = os.path.join(output_dir, f"block_{timestamp}.jsonl")

#         # In-memory structures
#         # request_id -> list of event dicts (preserves order of logging)
#         self.events_by_request: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

#         # block_id -> (creator_user_id, creator_request_id) when a block is created
#         self.block_creators: Dict[str, tuple] = {}

#         # Optional: block_id -> list of hit events (each event dict)
#         self.block_hits: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

#     def log_block_event(
#         self,
#         event_type: str,
#         user_id: str,
#         request_id: str,
#         block_id: str,
#         is_attacker: bool = False,
#         extra: Optional[Dict[str, Any]] = None
#         ) -> Dict[str, Any]:
#         """
#         Log an event (created/hit/etc.) to file and keep it in memory grouped by request_id.

#         :param event_type: e.g. "created" or "hit"
#         :param user_id: user who performed the action
#         :param request_id: request identifier for the user action
#         :param block_id: identifier of the block that was created/hit
#         :param is_attacker: optional boolean flag (kept for compatibility)
#         :param extra: optional dict with arbitrary extra fields to include
#         :return: the event dictionary that was logged
#         """
#         event = {
#             "timestamp": datetime.now().isoformat(),
#             "event": event_type,
#             "user_id": user_id,
#             "request_id": request_id,
#             "block_id": block_id,
#             "is_attacker": bool(is_attacker)
#         }

#         if extra:
#             # Merge safely (do not overwrite core fields)
#             for k, v in extra.items():
#                 if k not in event:
#                     event[k] = v

#         # Append to file (JSONL)
#         with open(self.filepath, "a", encoding="utf-8") as f:
#             f.write(json.dumps(event) + "\n")

#         # Store in-memory grouped by request_id
#         self.events_by_request[request_id].append(event)

#         # Update block creator mapping if this is a creation event
#         if event_type == "created":
#             self.block_creators[block_id] = (user_id, request_id)

#         # If this is a hit, store in block_hits for convenience
#         if event_type == "hit":
#             self.block_hits[block_id].append(event)

#         return event

#     # --- Convenience / query helpers ---

#     def get_events_for_request(self, request_id: str) -> List[Dict[str, Any]]:
#         """Return list of events for a given request_id (empty list if none)."""
#         return list(self.events_by_request.get(request_id, []))

#     def get_creator_for_block(self, block_id: str) -> Optional[tuple]:
#         """Return (creator_user_id, creator_request_id) for a block, or None if unknown."""
#         return self.block_creators.get(block_id)

#     def get_hits_for_block(self, block_id: str) -> List[Dict[str, Any]]:
#         """Return list of 'hit' events recorded for a block (empty list if none)."""
#         return list(self.block_hits.get(block_id, []))

#     def dump_in_memory_to_file(self, output_path: Optional[str] = None) -> str:
#         """
#         Write the current in-memory events_by_request to a JSON file (for snapshotting).
#         Returns the path written to.
#         """
#         if output_path is None:
#             output_path = self.filepath + ".snapshot.json"

#         # Create parent directory if needed
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)

#         # Compose a serializable dict
#         serializable = {
#             "events_by_request": self.events_by_request,
#             "block_creators": self.block_creators,
#             "block_hits": self.block_hits
#         }

#         # Convert defaultdict/list structures to normal dicts/lists
#         serializable_clean = {
#             "events_by_request": {k: v for k, v in serializable["events_by_request"].items()},
#             "block_creators": {k: v for k, v in serializable["block_creators"].items()},
#             "block_hits": {k: v for k, v in serializable["block_hits"].items()}
#         }

#         with open(output_path, "w", encoding="utf-8") as f:
#             json.dump(serializable_clean, f, indent=2, default=str)

#         return output_path


# # singleton
# block_event_logger = BlockEventLogger()

##############################################################################################################################################

# import json
# from collections import defaultdict
# from datetime import datetime
# import os

# class BlockEventLogger:
#     def __init__(self, output_dir=None):
#         """
#         output_dir: directory where log file will be written
#         """
#         # Default path if not provided
#         if output_dir is None:
#             output_dir = "/home/panagiotis.pennas/vllm.v1/vllm/panos_experiments/results/metrics/default_logs"

#         os.makedirs(output_dir, exist_ok=True)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.filepath = os.path.join(output_dir, f"block_{timestamp}.jsonl")

#         # Initialize tracking structures
#         self.block_creators = {}
#         self.hit_counts = defaultdict(int)
#         self.user_hits = defaultdict(int)
#         self.user_creates = defaultdict(int)
#         self.user_cross_hits = defaultdict(int)
#         self.user_blocks = defaultdict(set)


#     def log_block_event(self, event_type, user_id, request_id, block_id, attacker_id=False):
#         """
#         Logs an event to the JSONL file.
#         Automatically computes and appends metrics for all users.
#         """
#         entry = {
#             "event": event_type,
#             "user_id": user_id,
#             "request_id": request_id,
#             "block_id": block_id,
#             "is_attacker": attacker_id
#         }

#         # Write the event itself
#         with open(self.filepath, "a") as f:
#             f.write(json.dumps(entry) + "\n")

#         # --- Update internal state ---
#         if event_type == "created":
#             self.block_creators[block_id] = (user_id, request_id)
#             self.user_creates[user_id] += 1
#             self.user_blocks[user_id].add(block_id)

#         elif event_type == "hit":
#             self.handle_hit(entry)

#         # --- Compute metrics automatically ---
#         metrics_entries = self.compute_metrics_json()
#         with open(self.filepath, "a") as f:
#             for metrics in metrics_entries:
#                 f.write(json.dumps(metrics) + "\n")

#         return entry

#     def handle_hit(self, entry):
#         hitter = entry["user_id"]
#         block_id = entry["block_id"]

#         creator_info = self.block_creators.get(block_id)
#         if creator_info is not None:
#             creator_id, _ = creator_info

#             self.hit_counts[(entry["request_id"], hitter, creator_id)] += 1
#             self.user_hits[hitter] += 1
#             self.user_blocks[hitter].add(block_id)

#             # Cross-user hit
#             if hitter != creator_id:
#                 self.user_cross_hits[hitter] += 1
#         # else:
#         #     print(f"Block {block_id} hit by user {hitter}, creator unknown")

#     def compute_metrics_json(self):
#         """
#         Prepares metrics JSON entries for all users.
#         """
#         metrics_entries = []
#         all_users = set(list(self.user_hits.keys()) + list(self.user_creates.keys()))

#         for user in all_users:
#             hits = self.user_hits.get(user, 0)
#             creates = self.user_creates.get(user, 0)
#             hcr = hits / (hits + creates) if (hits + creates) > 0 else 0.0
#             cross_hits = self.user_cross_hits.get(user, 0)
#             diversity = len(self.user_blocks.get(user, set()))

#             metrics_entry = {
#                 "event": "metrics",
#                 "user_id": user,
#                 "hit_created_ratio": hcr,
#                 "cross_user_hit_count": cross_hits,
#                 "block_diversity": diversity
#             }
#             metrics_entries.append(metrics_entry)

#         return metrics_entries

# # --- Singleton instance ---
# block_event_logger = BlockEventLogger()






##############################################################################################################################################




# import json
# from collections import defaultdict
# from datetime import datetime
# import os

# class BlockEventLogger:
#     def __init__(self, output_dir=None):
#         """
#         output_dir: directory where log file will be written
#         """
#         if output_dir is None:
#             output_dir = "/home/panagiotis.pennas/vllm.v1/vllm/panos_experiments/results/metrics/default_logs"

#         os.makedirs(output_dir, exist_ok=True)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.filepath = os.path.join(output_dir, f"block_{timestamp}.jsonl")

#         # Initialize tracking structures
#         self.block_creators = {}
#         self.hit_counts = defaultdict(int)
#         self.user_hits = defaultdict(int)
#         self.user_creates = defaultdict(int)
#         self.user_cross_hits = defaultdict(int)
#         self.user_blocks = defaultdict(set)

#     def log_block_event(self, event_type, user_id, request_id, block_id, attacker_id=False):
#         """
#         Logs an event to the JSONL file and updates internal metrics.
#         """
#         entry = {
#             "event": event_type,
#             "user_id": user_id,
#             "request_id": request_id,
#             "block_id": block_id,
#             "is_attacker": attacker_id
#         }

#         with open(self.filepath, "a") as f:
#             f.write(json.dumps(entry) + "\n")

#         if event_type == "created":
#             self.block_creators[block_id] = (user_id, request_id)
#             self.user_creates[user_id] += 1
#             self.user_blocks[user_id].add(block_id)
#         elif event_type == "hit":
#             self.handle_hit(entry)

#         # Append metrics automatically
#         metrics_entries = self.compute_metrics_json()
#         with open(self.filepath, "a") as f:
#             for metrics in metrics_entries:
#                 f.write(json.dumps(metrics) + "\n")

#         return entry

#     def handle_hit(self, entry):
#         hitter = entry["user_id"]
#         block_id = entry["block_id"]

#         creator_info = self.block_creators.get(block_id)
#         if creator_info is not None:
#             creator_id, _ = creator_info

#             self.hit_counts[(entry["request_id"], hitter, creator_id)] += 1
#             self.user_hits[hitter] += 1
#             self.user_blocks[hitter].add(block_id)

#             if hitter != creator_id:
#                 self.user_cross_hits[hitter] += 1

#     def compute_metrics_json(self):
#         """
#         Returns a list of metrics entries for all users.
#         """
#         metrics_entries = []
#         all_users = set(list(self.user_hits.keys()) + list(self.user_creates.keys()))

#         for user in all_users:
#             hits = self.user_hits.get(user, 0)
#             creates = self.user_creates.get(user, 0)
#             hcr = hits / (hits + creates) if (hits + creates) > 0 else 0.0
#             cross_hits = self.user_cross_hits.get(user, 0)
#             diversity = len(self.user_blocks.get(user, set()))

#             metrics_entry = {
#                 "event": "metrics",
#                 "user_id": user,
#                 "hit_created_ratio": hcr,
#                 "cross_user_hit_count": cross_hits,
#                 "block_diversity": diversity
#             }
#             metrics_entries.append(metrics_entry)

#         return metrics_entries

# # --- Lazy singleton for global access ---
# _block_event_logger = None

# def get_block_event_logger(output_dir=None):
#     """
#     Returns a singleton BlockEventLogger.
#     Creates it if it doesn't exist yet, using the given output_dir.
#     """
#     global _block_event_logger
#     if _block_event_logger is None:
#         _block_event_logger = BlockEventLogger(output_dir=output_dir)
#     return _block_event_logger









##############################################################################################################################################








# import json
# from collections import defaultdict
# from datetime import datetime
# import os
# import joblib
# import warnings
# from sklearn.exceptions import InconsistentVersionWarning
# import pandas as pd

# # Suppress the sklearn version mismatch warning you saw earlier
# warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# class BlockEventLogger:
#     def __init__(self, output_dir=None, classifier_path=None):
#         """
#         output_dir: directory where log file will be written
#         classifier_path: path to a pre-trained attacker classifier
#         """
#         if output_dir is None:
#             output_dir = "/home/panagiotis.pennas/vllm.v1/vllm/panos_experiments/results/metrics/default_logs"

#         os.makedirs(output_dir, exist_ok=True)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.filepath = os.path.join(output_dir, f"block_{timestamp}.jsonl")

#         # Initialize tracking structures
#         self.block_creators = {}
#         self.hit_counts = defaultdict(int)
#         self.user_hits = defaultdict(int)
#         self.user_creates = defaultdict(int)
#         self.user_cross_hits = defaultdict(int)
#         self.user_blocks = defaultdict(set)

#         # Load pre-trained classifier (if provided)
#         if classifier_path and os.path.exists(classifier_path):
#             try:
#                 self.classifier = joblib.load(classifier_path)
#             except Exception as e:
#                 # If loading fails, fall back to no classifier but log the error to console
#                 print(f"[WARN] Failed to load classifier from {classifier_path}: {e}")
#                 self.classifier = None
#         else:
#             self.classifier = None

#     def log_block_event(self, event_type, user_id, request_id, block_id):
#         """
#         Logs a single unified event record with metrics and attacker info.
#         """
#         # Update internal state first
#         if event_type == "created":
#             self.block_creators[block_id] = (user_id, request_id)
#             self.user_creates[user_id] += 1
#             self.user_blocks[user_id].add(block_id)
#         elif event_type == "hit":
#             self.handle_hit({
#                 "event": event_type,
#                 "user_id": user_id,
#                 "request_id": request_id,
#                 "block_id": block_id
#             })

#         # Compute metrics for this user
#         metrics_entries = self.compute_metrics_json()
#         user_metrics = next((m for m in metrics_entries if m["user_id"] == user_id), {})

#         # Predict attacker
#         is_attacker, attacker_score = self.predict_attacker(user_id, return_score=True)

#         # Combine everything into one record
#         record = {
#             "event": event_type,
#             "user_id": user_id,
#             "request_id": request_id,
#             "block_id": block_id,
#             "hit_created_ratio": user_metrics.get("hit_created_ratio", 0.0),
#             "cross_user_hit_count": user_metrics.get("cross_user_hit_count", 0),
#             "block_diversity": user_metrics.get("block_diversity", 0),
#             "is_attacker": is_attacker,
#             "attacker_score": attacker_score
#         }

#         # Write the unified record
#         with open(self.filepath, "a") as f:
#             f.write(json.dumps(record) + "\n")

#         # Optional: print alert if attacker
#         if is_attacker:
#             print(f"[ALERT] User {user_id} flagged as attacker (score={attacker_score})")

#         return record

#     def handle_hit(self, entry):
#         hitter = entry["user_id"]
#         block_id = entry["block_id"]

#         creator_info = self.block_creators.get(block_id)
#         if creator_info is not None:
#             creator_id, _ = creator_info

#             self.hit_counts[(entry["request_id"], hitter, creator_id)] += 1
#             self.user_hits[hitter] += 1
#             self.user_blocks[hitter].add(block_id)

#             if hitter != creator_id:
#                 self.user_cross_hits[hitter] += 1

#     def compute_metrics_json(self):
#         """
#         Returns a list of metrics entries for all users.
#         """
#         metrics_entries = []
#         all_users = set(list(self.user_hits.keys()) + list(self.user_creates.keys()))

#         for user in all_users:
#             hits = self.user_hits.get(user, 0)
#             creates = self.user_creates.get(user, 0)
#             hcr = hits / (hits + creates) if (hits + creates) > 0 else 0.0
#             cross_hits = self.user_cross_hits.get(user, 0)
#             diversity = len(self.user_blocks.get(user, set()))

#             metrics_entry = {
#                 "event": "metrics",
#                 "user_id": user,
#                 "hit_created_ratio": hcr,
#                 "cross_user_hit_count": cross_hits,
#                 "block_diversity": diversity
#             }
#             metrics_entries.append(metrics_entry)

#         return metrics_entries

#     def predict_attacker(self, user_id, return_score: bool = False):
#         """
#         Predicts if a user is an attacker using the loaded classifier.
#         Returns (bool_is_attacker, score_or_none) if return_score True, else just bool.
#         Uses a DataFrame with the same feature names to avoid sklearn warnings.
#         """
#         if self.classifier is None:
#             return (False, None) if return_score else False

#         metrics = self.compute_metrics_json()
#         user_metrics = next((m for m in metrics if m["user_id"] == user_id), None)
#         if user_metrics is None:
#             return (False, None) if return_score else False

#         features = ["hit_created_ratio", "block_diversity", "cross_user_hit_count"]
#         # Build DataFrame with column names to match training
#         X_user = pd.DataFrame([{f: user_metrics[f] for f in features}])

#         try:
#             pred = self.classifier.predict(X_user)[0]
#         except Exception as e:
#             # If prediction fails for any reason, return False and log warning
#             print(f"[WARN] classifier.predict failed for user {user_id}: {e}")
#             return (False, None) if return_score else False

#         score = None
#         # If classifier supports predict_proba, give the positive-class probability
#         if hasattr(self.classifier, "predict_proba"):
#             try:
#                 proba = self.classifier.predict_proba(X_user)[0]
#                 # assume positive class is at index 1 (common for binary models)
#                 if len(proba) > 1:
#                     score = float(proba[1])
#                 else:
#                     score = float(proba[0])
#             except Exception:
#                 score = None

#         if return_score:
#             return (bool(pred), score)
#         else:
#             return bool(pred)


# # --- Lazy singleton for global access ---
# _block_event_logger = None

# def get_block_event_logger(output_dir=None):
#     """
#     Returns a singleton BlockEventLogger.
#     Creates it if it doesn't exist yet, using the given output_dir.
#     """
#     global _block_event_logger
#     if _block_event_logger is None:
#         _block_event_logger = BlockEventLogger(
#             output_dir=output_dir,
#             classifier_path="/home/panagiotis.pennas/vllm.v1/vllm/panos_experiments/classifiers/xgb_model.joblib"
#         )
#     return _block_event_logger


##############################################################################################################################################


# import json
# from collections import defaultdict
# from datetime import datetime
# import os
# import joblib
# import warnings
# from sklearn.exceptions import InconsistentVersionWarning
# import pandas as pd
# import numpy as np
# import xgboost as xgb

# warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# class BlockEventLogger:
#     def __init__(self, output_dir=None, classifier_path=None, feature_columns=None):
#         if output_dir is None:
#             output_dir = "./logs"
#         os.makedirs(output_dir, exist_ok=True)

#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.filepath = os.path.join(output_dir, f"{timestamp}_block.jsonl")

#         # Internal trackers
#         self.block_creators = {}
#         self.user_hits = defaultdict(int)
#         self.user_creates = defaultdict(int)
#         self.user_cross_hits = defaultdict(int)
#         self.user_blocks = defaultdict(set)
#         self.request_events = defaultdict(list)

#         # Features used for prediction
#         self.feature_columns = feature_columns or [
#             "hit_created_ratio",
#             "cross_user_hit_count",
#             "block_diversity",
#             "n_events",
#             "n_hit",
#             "n_created",
#             "frac_hit",
#         ]

#         # Load classifier safely using XGBoost's load_model
#         self.classifier = None
#         if classifier_path and os.path.exists(classifier_path):
#             try:
#                 print(f"[INFO] Loading XGBoost classifier from {classifier_path} (CPU-only)...")
#                 self.classifier = xgb.XGBClassifier(tree_method='hist', predictor='cpu_predictor')
#                 self.classifier.load_model(classifier_path)  # <- safe CPU load

#                 # Ensure CPU mode
#                 if hasattr(self.classifier, "tree_method") and self.classifier.tree_method == "gpu_hist":
#                     print("[INFO] Switching XGBoost model to CPU mode...")
#                     self.classifier.tree_method = "hist"
#                 if hasattr(self.classifier, "gpu_id"):
#                     self.classifier.gpu_id = -1
#                 if hasattr(self.classifier, "use_label_encoder"):
#                     self.classifier.use_label_encoder = False

#                 print(f"[INFO] Classifier loaded successfully.")
#             except Exception as e:
#                 print(f"[WARN] Failed to load classifier: {e}. Running in metrics-only mode.")
#                 self.classifier = None
#         else:
#             print("[INFO] No classifier loaded. Running in metrics-only mode.")


#     # --- Event Logging -------------------------------------------------------
#     def log_block_event(self, event_type, user_id, request_id, block_id):
#         """
#         Logs a single block event (created or hit) and performs a real-time attacker prediction.
#         """
#         # Update in-memory structures
#         if event_type == "created":
#             self.block_creators[block_id] = (user_id, request_id)
#             self.user_creates[user_id] += 1
#             self.user_blocks[user_id].add(block_id)
#         elif event_type == "hit":
#             self.handle_hit(user_id, block_id, request_id)

#         # Record this event for the request-level history
#         self.request_events[request_id].append({
#             "event": event_type,
#             "user_id": user_id,
#             "block_id": block_id,
#             "timestamp": datetime.now().isoformat(),
#         })

#         # Compute per-user metrics
#         user_metrics = self.compute_user_metrics(user_id)

#         # Compute per-request metrics (if enough data)
#         request_metrics = self.compute_request_features(request_id)

#         # Combine all features for prediction
#         all_features = {**user_metrics, **request_metrics}
#         X_live = pd.DataFrame([{k: all_features.get(k, 0) for k in self.feature_columns}])

#         # Predict attacker
#         is_attacker, score = self.predict_attacker(X_live)

#         # Log record
#         record = {
#             "event": event_type,
#             "user_id": user_id,
#             "request_id": request_id,
#             "block_id": block_id,
#             "is_attacker": is_attacker,
#             "attacker_score": score,
#             **all_features
#         }
#         with open(self.filepath, "a") as f:
#             f.write(json.dumps(record) + "\n")

#         if is_attacker:
#             print(f"[ALERT] User {user_id} flagged as attacker (score={score:.3f})")

#         return record

#     def handle_hit(self, user_id, block_id, request_id):
#         creator_info = self.block_creators.get(block_id)
#         if creator_info:
#             creator_id, _ = creator_info
#             self.user_hits[user_id] += 1
#             self.user_blocks[user_id].add(block_id)
#             if user_id != creator_id:
#                 self.user_cross_hits[user_id] += 1

#     def compute_user_metrics(self, user_id):
#         """
#         Compute per-user aggregate statistics (matching training features).
#         """
#         hits = self.user_hits[user_id]
#         creates = self.user_creates[user_id]
#         cross_hits = self.user_cross_hits[user_id]
#         diversity = len(self.user_blocks[user_id])

#         hcr = hits / (hits + creates) if (hits + creates) > 0 else 0.0

#         return {
#             "hit_created_ratio": hcr,
#             "cross_user_hit_count": cross_hits,
#             "block_diversity": diversity,
#         }

#     def compute_request_features(self, request_id):
#         """
#         Compute per-request aggregated features based on past events in this request.
#         """
#         events = self.request_events[request_id]
#         n_events = len(events)
#         n_hit = sum(1 for e in events if e["event"] == "hit")
#         n_created = sum(1 for e in events if e["event"] == "created")
#         frac_hit = n_hit / n_events if n_events else 0

#         return {
#             "n_events": n_events,
#             "n_hit": n_hit,
#             "n_created": n_created,
#             "frac_hit": frac_hit,
#         }

#     def predict_attacker(self, X_live):
#         """
#         Use the loaded model to predict if the current behavior indicates an attacker.
#         """
#         if self.classifier is None:
#             return False, 0.0

#         try:
#             proba = self.classifier.predict_proba(X_live)[0]
#             score = float(proba[1]) if len(proba) > 1 else float(proba[0])
#             return (score > 0.5, score)
#         except Exception as e:
#             print(f"[WARN] Prediction failed: {e}")
#             return False, 0.0


# _block_event_logger = None

# def get_block_event_logger(output_dir=None):
#     global _block_event_logger
#     if _block_event_logger is None:
#         _block_event_logger = BlockEventLogger(
#             output_dir=output_dir,
#             classifier_path="/home/panagiotis.pennas/vllm.v1/vllm/panos_experiments/workloads/datasets/.json"
#         )
#     return _block_event_logger
    # pass







##############################################################################################################################################



# detector_per_request_with_features.py
# detector_per_request_with_features.py
# import os
# import json
# from collections import defaultdict
# from datetime import datetime

# import pandas as pd
# import numpy as np
# import joblib


# class RequestEventLogger:
#     def __init__(
#         self,
#         model_path="/home/panagiotis.pennas/vllm.v1/vllm/panos_experiments/classifiers/attack_detector_per_request.pkl",
#         output_dir=None,
#         detection_threshold=0.9
#     ):
#         if output_dir is None:
#             output_dir = "/home/panagiotis.pennas/vllm.v1/vllm/panos_experiments/results/metrics/realtime_logs"

#         os.makedirs(output_dir, exist_ok=True)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.filepath = os.path.join(output_dir, f"request_live_{timestamp}.jsonl")

#         # Load classifier
#         try:
#             loaded = joblib.load(model_path)
#             if isinstance(loaded, tuple) and len(loaded) == 2:
#                 self.model, self.feature_cols = loaded
#             else:
#                 raise RuntimeError("Unexpected model artifact shape")
#             print(f"✅ Per-request classifier loaded successfully from: {model_path}")
#         except Exception as e:
#             raise RuntimeError(f"❌ Failed to load classifier from {model_path}: {e}")

#         self.pending_requests = defaultdict(list)
#         self.threshold = detection_threshold
#         self.flagged_requests = set()

#     # ------------------------------
#     # vLLM-compatible interface
#     # ------------------------------
#     def log_block_event(self, event_type, user_id, request_id, block_id, **kwargs):
#         if request_id in self.flagged_requests:
#             return None

#         event = {
#             "event": event_type,
#             "user_id": user_id,
#             "request_id": request_id,
#             "block_id": block_id
#         }
#         event.update(kwargs)

#         self.pending_requests[request_id].append(event)

#         # Compute features and detect
#         prob, flagged = self.detect_request(request_id)
#         if flagged:
#             self.flagged_requests.add(request_id)
#             return {"request_id": request_id, "probability": prob, "events": self.pending_requests[request_id]}
#         return None

#     # ------------------------------
#     # Feature computation per request
#     # ------------------------------
#     def detect_request(self, request_id):
#         events = self.pending_requests[request_id]
#         df = pd.DataFrame(events)

#         # Use the first user_id as representative for logging
#         user_id = df["user_id"].iloc[0] if not df.empty else "unknown"

#         n_hit = (df["event"] == "hit").sum()
#         n_created = (df["event"] == "created").sum()
#         n_events = len(df)
#         hit_created_ratio = n_hit / n_created if n_created > 0 else 0.0
#         frac_hit = n_hit / n_events if n_events > 0 else 0.0
#         cross_user_hit_count = ((df["event"] == "hit") & (df["user_id"] != df.get("block_creator_id", df["user_id"]))).sum()
#         block_diversity = df["block_id"].nunique()
#         event_diversity = df["event"].nunique()

#         # Construct feature row
#         feature_row = pd.DataFrame([{
#             "n_hit": n_hit,
#             "n_created": n_created,
#             "n_events": n_events,
#             "hit_created_ratio": hit_created_ratio,
#             "frac_hit": frac_hit,
#             "cross_user_hit_count": cross_user_hit_count,
#             "block_diversity": block_diversity,
#             "event_diversity": event_diversity
#         }])[self.feature_cols].fillna(0)

#         # Apply classifier
#         prob = float(self.model.predict_proba(feature_row)[0, 1])
#         flagged = prob > self.threshold

#         # Log features and result (user_id included)
#         log_entry = feature_row.iloc[0].to_dict()
#         log_entry.update({"attack_prob": prob, "flagged": flagged, "request_id": request_id, "user_id": user_id})

#         with open(self.filepath, "a") as f:
#             f.write(json.dumps(log_entry) + "\n")

#         return prob, flagged


# # ------------------------------
# # Singleton
# # ------------------------------
# _request_event_logger = None

# def get_block_event_logger(output_dir=None, **kwargs):
#     global _request_event_logger
#     if _request_event_logger is None:
#         _request_event_logger = RequestEventLogger(output_dir=output_dir, **kwargs)
#     return _request_event_logger


# import os
# import json
# from datetime import datetime
# from collections import defaultdict
# import pandas as pd  # keep if you still want to analyze events later

# class RequestEventLogger:
#     def __init__(self, output_dir=None):
#         if output_dir is None:
#             output_dir = "/home/panagiotis.pennas/vllm.v1/vllm/panos_experiments/results/metrics/realtime_logs"

#         os.makedirs(output_dir, exist_ok=True)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.filepath = os.path.join(output_dir, f"request_live_{timestamp}.jsonl")

#         self.pending_requests = defaultdict(list)
#         self.logged_requests = set()

#         print(f"📝 Event logger initialized. Logging to: {self.filepath}")

#     # ------------------------------
#     # vLLM-compatible interface
#     # ------------------------------
#     def log_block_event(self, event_type, user_id, request_id, block_id, **kwargs):
#         """Store block-level event data and write to file when request completes."""
#         if request_id in self.logged_requests:
#             return None  # already logged this request

#         event = {
#             "event": event_type,
#             "user_id": user_id,
#             "request_id": request_id,
#             "block_id": block_id,
#             "timestamp": datetime.now().isoformat()
#         }
#         event.update(kwargs)

#         # Store event in memory
#         self.pending_requests[request_id].append(event)

#         # For demonstration, you could flush immediately after each event
#         # or only after certain condition (e.g., 'request_end' event)
#         self._flush_event(event)
#         return None

#     # ------------------------------
#     # Write event to file
#     # ------------------------------
#     def _flush_event(self, event):
#         """Append a single event to the JSONL file."""
#         with open(self.filepath, "a") as f:
#             f.write(json.dumps(event) + "\n")

#     # ------------------------------
#     # Optional: flush all stored events (if needed)
#     # ------------------------------
#     def flush_all(self):
#         """Write all pending events to disk (if batching was used)."""
#         with open(self.filepath, "a") as f:
#             for req_id, events in self.pending_requests.items():
#                 for event in events:
#                     f.write(json.dumps(event) + "\n")
#         self.pending_requests.clear()
#         print(f"✅ All pending events flushed to {self.filepath}")

# # ------------------------------
# # Singleton pattern
# # ------------------------------
# _request_event_logger = None

# def get_block_event_logger(output_dir=None):
#     """Return a global singleton instance of the logger."""
#     global _request_event_logger
#     if _request_event_logger is None:
#         _request_event_logger = RequestEventLogger(output_dir=output_dir)
#     return _request_event_logger



