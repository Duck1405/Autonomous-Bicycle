import json
import os
from datetime import datetime, timezone

import numpy as np
import torch


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def append_jsonl(path, record):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        'logged_at_utc': datetime.now(timezone.utc).isoformat(),
        **to_jsonable(record),
    }
    with open(path, 'a') as f:
        f.write(json.dumps(payload, sort_keys=True) + '\n')
