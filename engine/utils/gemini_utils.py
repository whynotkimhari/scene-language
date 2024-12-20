from engine.constants import PROJ_DIR, GOOGLE_API_KEY, MAX_TOKENS, TEMPERATURE, NUM_COMPLETIONS
import copy
from pathlib import Path
import base64
import json
import os
import time
import random
from PIL import Image
import io
import google
import google.generativeai as genai


GOOGLE_MODEL_NAME = 'gemini-1.5-flash'

class GeminiClient:
    def __init__(self, model_name=GOOGLE_MODEL_NAME, cache="cache.json"):
        self.cache_file = cache
        self.model_name = model_name
        self.exponential_backoff = 1
        # Load the cache JSON file, if cache file exists. Else, cache is {}
        if os.path.exists(cache):
            while os.path.exists(self.cache_file + ".tmp") or os.path.exists(self.cache_file + ".lock"):
                time.sleep(0.1)
            with open(cache, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

        genai.configure(api_key=GOOGLE_API_KEY)

    def generate(self, user_prompt, system_prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, stop_sequences=None, verbose=False,
                 num_completions=NUM_COMPLETIONS, skip_cache_completions=0, skip_cache=False):

        print(f'[INFO] Gemini: querying for {num_completions=}, {skip_cache_completions=}')
        if skip_cache:
            print(f'[INFO] Gemini: Skipping cache')
        if verbose:
            print(user_prompt)
            print("-----")

        # Prepare messages for the API request
        if isinstance(user_prompt, str):
            content = [{"type": "text", "text": user_prompt}]
        elif isinstance(user_prompt, list):
            content = []
            for content_item in user_prompt:
                if content_item['type'] == 'image_url' and os.path.exists(content_item['image_url']):
                    with open(content_item['image_url'], "rb") as f:
                        image_data = base64.b64encode(f.read()).decode("utf-8")
                    content_item = {
                        'type': 'image',
                        "source": {
                            "type": "base64",
                            "media_type": f'image/{Path(content_item["image_url"]).suffix.removeprefix(".")}',
                            "data": image_data,
                        },
                    }
                content.append(content_item)
        else:
            raise RuntimeError(user_prompt)
        messages = [{"role": "user", "content": content}]

        cache_key = None
        results = []
        if not skip_cache:
            cache_key = str((user_prompt, system_prompt, max_tokens, temperature, stop_sequences, 'gemini'))

            num_completions = skip_cache_completions + num_completions
            if cache_key in self.cache:
                print(f'[INFO] Gemini: cache hit {len(self.cache[cache_key])}')
                if len(self.cache[cache_key]) < num_completions:
                    num_completions -= len(self.cache[cache_key])
                    results = self.cache[cache_key]
                else:
                    return cache_key, self.cache[cache_key][skip_cache_completions:num_completions]

        while num_completions > 0:
            self.client = genai.GenerativeModel(
                self.model_name,
                system_instruction=system_prompt
            )
            while True:
                try:
                    response = self.client.generate_content(messages)
                    break
                except Exception:
                    print("Rate limit reached. Waiting before retrying...")
                    time.sleep(16 * self.exponential_backoff)
                    self.exponential_backoff *= 2
            num_completions -= 1

            content = []
            if response.text:
                for text_block in response.text:
                    content.append(text_block.text)

            indented = []
            for c in content:
                indented.append(c.split('\n'))
            results.extend(indented)

        if not skip_cache:
            self.update_cache(cache_key, results)

        return cache_key, results[skip_cache_completions:]

    def update_cache(self, cache_key, results):
        while os.path.exists(self.cache_file + ".tmp") or os.path.exists(self.cache_file + ".lock"):
            time.sleep(0.1)
        with open(self.cache_file + ".lock", "w") as f:
            pass
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                self.cache = json.load(f)
        self.cache[cache_key] = results
        with open(self.cache_file + ".tmp", "w") as f:
            json.dump(self.cache, f)
        os.rename(self.cache_file + ".tmp", self.cache_file)
        os.remove(self.cache_file + ".lock")


def setup_gemini():
    try:
        username = os.getlogin()
    except OSError:
        username = os.environ.get('USER') or os.environ.get('LOGNAME')

    model = GeminiClient(cache='cache.json' if not os.path.exists('/viscam/') else f'cache_{username}.json')
    return model