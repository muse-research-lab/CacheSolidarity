from typing import List, Tuple
import time
from collections import defaultdict
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

########################################################### User Isolation ###########################################################

class UserMonitor:
    def __init__(self):
        # Keep a log of which users sent requests
        self.user_prompts = defaultdict(list)

    def record_request(self, uid: str, prompt: str):
        self.user_prompts[uid].append(prompt)
        # print(f"[INFO] Request recorded from user={uid}")
    # pass



########################################################## Similarity-based Attack Detection ###########################################################

# class UserMonitor:
    # def __init__(self):
    #     self.user_prompts: defaultdict[str, List[Tuple[float, str]]] = defaultdict(list)
    #     self.attack_flags: dict[str, bool] = {}
    #     self.prompt_embeddings: dict[str, any] = {}  # Cache for prompt embeddings

    # def record_request(self, uid: str, prompt: str):
    #     self.user_prompts[uid].append((time.time(), prompt))
    #     self.check_attack(uid)

    # def is_attacker(self, uid: str) -> bool:
    #     return self.attack_flags.get(uid, False)

    # def get_embedding(self, prompt: str):
    #     # Use cached embedding if available
    #     if prompt in self.prompt_embeddings:
    #         return self.prompt_embeddings[prompt]
    #     # Compute and cache embedding
    #     vectorizer = TfidfVectorizer().fit([prompt])
    #     vector = vectorizer.transform([prompt])
    #     self.prompt_embeddings[prompt] = (vectorizer, vector)
    #     return (vectorizer, vector)

    # def check_attack(self, uid: str):
    #     prompt_data = self.user_prompts[uid]
    #     prompts = [p for _, p in prompt_data]

    #     for i in range(len(prompts)):
    #         for j in range(i + 1, len(prompts)):
    #             p1, p2 = prompts[i], prompts[j]

    #             # Get embeddings (vectorizer, vector) for both prompts
    #             v1, vec1 = self.get_embedding(p1)
    #             v2, vec2 = self.get_embedding(p2)

    #             # To compare, need to use a common vectorizer (fit on both prompts)
    #             vectorizer = TfidfVectorizer().fit([p1, p2])
    #             vectors = vectorizer.transform([p1, p2])
    #             cos_sim = cosine_similarity(vectors[0], vectors[1])[0][0]

    #             if cos_sim > 0.8:
    #                 self.attack_flags[uid] = True
    #                 return
    # pass








########################################################### Length-based Attack Detection ###########################################################

# class UserMonitor:
#     def __init__(self):
#         self.last_prompt_length: dict[str, int] = {}
#         self.last_prompt_user: dict[int, str] = {}  # Maps length to user id
#         self.attack_flags: dict[str, bool] = {}

#     def record_request(self, uid: str, prompt: str):
#         length = len(prompt)
#         # Check if another user has sent a prompt with the same length
#         if length in self.last_prompt_user and self.last_prompt_user[length] != uid:
#             other_uid = self.last_prompt_user[length]
#             print(f"[ALERT] User '{uid}' sent a prompt with the same length ({length}) as user '{other_uid}'")
#             self.attack_flags[uid] = True
#         else:
#             self.attack_flags[uid] = False
#         # Update last prompt info
#         self.last_prompt_length[uid] = length
#         self.last_prompt_user[length] = uid

#     def is_attacker(self, uid: str) -> bool:
#         return self.attack_flags.get(uid, False)












