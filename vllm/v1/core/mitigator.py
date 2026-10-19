import time
import logging
from vllm.v1.request import Request
from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = logging.getLogger(__name__)

class KVCacheMitigator:
    """
    Handles mitigation for requests hitting blocks under attack.
    """

    def __init__(self, sleep_fn: callable = None):

        self.sleep_fn = sleep_fn or time.sleep

    def compute_delay(self, request: Request) -> float:
        
        rps = 30
        x = request.num_tokens

        if rps == 1:
            add_time = 0.000167 * x + 0.023982
        elif rps == 2:
            add_time = 0.000142 * x + 0.027098
        elif rps == 3:
            add_time = 0.000123 * x + 0.027855
        elif rps == 4:
            add_time = 0.000113 * x + 0.027195
        elif rps == 5:
            add_time = 0.000110 * x + 0.025669
        elif rps == 6:
            add_time = 0.000099 * x + 0.026182
        elif rps == 7:
            add_time = 0.000106 * x + 0.025831
        elif rps == 8:
            add_time = 0.000102 * x + 0.025706
        elif rps == 9:
            add_time = 0.000107 * x + 0.022984
        elif rps == 10:
            add_time = 0.000109 * x + 0.023112
        elif rps == 11:
            add_time = 0.000120 * x + 0.021444
        elif rps == 12:
            add_time = 0.000124 * x + 0.021081
        elif rps == 13:
            add_time = 0.000127 * x + 0.021233
        elif rps == 14:
            add_time = 0.000132 * x + 0.020848
        elif rps == 15:
            add_time = 0.000149 * x + 0.019884
        elif rps == 17:
            add_time = 0.000161 * x + 0.018062
        elif rps == 18:
            add_time = 0.000179 * x + 0.015381
        elif rps == 19:
            add_time = 0.000186 * x + 0.015503
        elif rps == 20:
            add_time = 0.000205 * x + 0.012565
        elif rps == 21:
            add_time = 0.000187 * x + 0.017224
        elif rps == 22:
            add_time = 0.000201 * x + 0.016071
        elif rps == 23:
            add_time = 0.000207 * x + 0.015104
        elif rps == 24:
            add_time = 0.000287 * x + 0.005077
        elif rps == 25:
            add_time = 0.000302 * x + 0.005242
        elif rps == 26:
            add_time = 0.000324 * x + 0.001071
        elif rps == 27:
            add_time = 0.000308 * x + 0.004508
        elif rps == 28:
            add_time = 0.000368 * x + 0.000008
        elif rps == 30:
            add_time = 0.000397 * x - 0.003957
        else:
            raise ValueError(f"No TTFT model available for RPS = {rps}")


        delay = add_time + time.monotonic()
        request.scheduled_delay = delay
