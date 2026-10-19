# defender.py
from typing import Dict


class Defender:
    def __init__(self):
        self.user_request_count: Dict[str, int] = {}
        self.detected_attackers: Dict[str, int] = {}
        self.user_max_cross_hits: Dict[str, Dict[str, int]] = {} 
        self.user_increase_count: Dict[str, Dict[str, int]] = {}  

    def evaluate_user(self, user_id: str, metrics: Dict[str, float]) -> bool:
        self.user_request_count[user_id] = self.user_request_count.get(user_id, 0) + 1
        req_number = self.user_request_count[user_id]

        if user_id in self.detected_attackers:
            return False

        v = {k: float(metrics.get(k, 0)) for k in [
            "variance_events",
            "variance_hits",
            "variance_new",
            "variance_own_cache_hit_rate",
            "variance_cross_cache_hit_rate",
        ]}
        cross_user_hits = metrics.get("cross_user_hits") or {}

        detected = False

        if req_number > 1:
            if v["variance_events"] < 0.5:
                detected = True
            elif all(val == 0 for val in v.values()):
                detected = True
            elif (
                v["variance_events"] == 0
                and v["variance_hits"] != 0
                and v["variance_new"] != 0
                and v["variance_own_cache_hit_rate"] == 0
            ):
                detected = True

        if not detected and isinstance(cross_user_hits, dict) and cross_user_hits:
            user_max_hits = self.user_max_cross_hits.setdefault(user_id, {})
            user_increase_count = self.user_increase_count.setdefault(user_id, {})

            for other_user, hit_count in cross_user_hits.items():
                prev_max = user_max_hits.get(other_user, 0)
                print(f"User {user_id} hit {hit_count} blocks from {other_user} (prev max: {prev_max})")

                if hit_count > prev_max:
                    user_max_hits[other_user] = hit_count
                    user_increase_count[other_user] = user_increase_count.get(other_user, 0) + 1

                    if user_increase_count[other_user] >= 2:
                        detected = True
                        break

        if detected:
            self.detected_attackers[user_id] = req_number
            print(f"Attacker detected: {user_id} at request #{req_number}")
            self._print_detected_summary()

        return detected

    def _print_detected_summary(self):
        """Print all detected attackers so far."""
        if not self.detected_attackers:
            print("No attackers detected yet.")
        else:
            print("\nDetected attackers so far:")
            for uid, req in self.detected_attackers.items():
                print(f"   - {uid} (detected at request #{req})")
