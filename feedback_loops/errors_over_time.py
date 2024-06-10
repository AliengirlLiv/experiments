import time
from concurrent.futures import ThreadPoolExecutor
import anthropic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


def load_cache():
    with open('errors_over_time_cache.py', 'r') as f:
        return json.loads(f.read())

def query_llm_batched(messages, max_tokens):
    response = anthropic.messages.create(
        model='claude-1.2-instant',
        max_tokens=max_tokens,
        messages=messages,
        temperature=0,
        # claude API: https://docs.anthropic.com/claude/reference/messages_post
    ).content[0].text
    return response
    