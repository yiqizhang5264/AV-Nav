"""Explicit multimodal verification; prompts are reconstruction choices."""
import base64
import io
import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from PIL import Image, ImageDraw


def parse_reply(text, kind):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('VLM returned no final JSON content')
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Qwen may add explanatory prose before its fenced final answer.
        # Accept exactly one complete object, never infer a result from prose.
        decoder = json.JSONDecoder()
        objects = []
        cursor = 0
        while cursor < len(text):
            start = text.find('{', cursor)
            if start < 0:
                break
            try:
                item, consumed = decoder.raw_decode(text[start:])
                objects.append(item)
                cursor = start + consumed
            except json.JSONDecodeError:
                cursor = start + 1
        if len(objects) != 1:
            raise ValueError('Expected exactly one complete final JSON object')
        value = objects[0]
    if not isinstance(value, dict):
        raise ValueError('Expected a final JSON object')
    keys = ('visibility', 'perspective') if kind == 'sufficiency' else ('matches',)
    for key in keys:
        if kind == 'sufficiency':
            if type(value.get(key)) is not int or not 1 <= value[key] <= 5:
                raise ValueError('Expected integer visibility/perspective scores in [1,5]')
        elif type(value.get(key)) is not bool:
            raise ValueError('Expected boolean matches')
    return {k: value[k] for k in keys}


class Verifier:
    def __init__(self, config):
        self.config = config

    def ask(self, observation, target, kind):
        image = Image.fromarray(observation.rgb)
        crop = image.crop(observation.bbox)
        if kind == 'sufficiency':
            prompt = (f'Target category: {target}. Image 1 is the full view; image 2 is the candidate crop. '
                      'Evaluate whether this candidate can be reliably identified. Score visibility '
                      '(clear, large enough, unoccluded) and perspective (identifying details visible) '
                      'independently from 1 (very poor) to 5 (excellent). Do not decide category membership yet. '
                      'Return JSON only: {"visibility": integer, "perspective": integer}.')
            images = [image, crop]
        else:
            ImageDraw.Draw(image).rectangle(observation.bbox, outline='red', width=3)
            prompt = (f'Does the object highlighted by the red rectangle belong to category "{target}"? '
                      'Use the highlighted object, not other objects in the scene. '
                      'Return JSON only: {"matches": true} or {"matches": false}.')
            images = [image]
        content = [{'type': 'text', 'text': prompt}]
        for image in images:
            stream = io.BytesIO()
            image.save(stream, format='PNG')
            encoded = base64.b64encode(stream.getvalue()).decode('ascii')
            content.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+encoded}})
        payload = dict(model=self.config['model'], messages=[dict(role='user', content=content)],
                       temperature=0, max_tokens=self.config['max_tokens'])
        payload['chat_template_kwargs'] = {'enable_thinking': not self.config.get('disable_thinking', False)}
        request = urllib.request.Request(self.config['base_url'].rstrip('/')+'/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json',
                'Authorization': 'Bearer '+os.environ.get('SAP_VLM_API_KEY', 'local')})
        failures = []
        for attempt in range(3):
            # Thinking can consume the initial budget before producing final JSON.
            # Preserve thinking and allow more completion tokens on retries.
            payload['max_tokens'] = min(12288, self.config['max_tokens'] + attempt * 2048)
            request.data = json.dumps(payload).encode()
            raw = None
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    raw = json.load(response)
                message = raw['choices'][0]['message']
                reply = message.get('content')
                result = parse_reply(reply, kind)
                return result, dict(prompt=prompt, response=reply, usage=raw.get('usage'),
                    reasoning=message.get('reasoning_content', message.get('reasoning')),
                    retry_failures=failures, completion_budget=payload['max_tokens'])
            except (ValueError, KeyError, IndexError, urllib.error.URLError, TimeoutError) as error:
                failure = dict(attempt=attempt + 1, kind=kind, error=str(error),
                               completion_budget=payload['max_tokens'], raw_response=raw)
                failures.append(failure)
                if os.environ.get('AV_RUN_DIR'):
                    with (Path(os.environ['AV_RUN_DIR'])/'vlm_errors.jsonl').open('a') as stream:
                        stream.write(json.dumps(failure)+'\n')
        raise RuntimeError('VLM verification failed after three attempts: '+failures[-1]['error'])
