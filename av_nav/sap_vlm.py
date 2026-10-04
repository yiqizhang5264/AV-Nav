"""Explicit multimodal verification; prompts are reconstruction choices."""
import base64
import io
import json
import os
import urllib.request
from PIL import Image, ImageDraw


def parse_reply(text, kind):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    value = json.loads(text)
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
        if self.config.get('disable_thinking'):
            payload['chat_template_kwargs'] = {'enable_thinking': False}
        request = urllib.request.Request(self.config['base_url'].rstrip('/')+'/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json',
                'Authorization': 'Bearer '+os.environ.get('SAP_VLM_API_KEY', 'local')})
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = json.load(response)
        reply = raw['choices'][0]['message']['content']
        return parse_reply(reply, kind), dict(prompt=prompt, response=reply, usage=raw.get('usage'))
