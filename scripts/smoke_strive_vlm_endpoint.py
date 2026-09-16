#!/usr/bin/env python3
"""Exercise STRIVE-style text and image structured outputs on a VLM endpoint."""

from __future__ import annotations

import argparse
import base64
import io

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel


class RoomChoice(BaseModel):
    final_answer: int
    reason: str


class Verification(BaseModel):
    flag: bool
    label: str


def image_data_url() -> str:
    image = Image.new("RGB", (32, 32), (255, 0, 0))
    data = io.BytesIO()
    image.save(data, format="JPEG")
    encoded = base64.b64encode(data.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--disable-thinking", action="store_true")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    extra_body = (
        {"chat_template_kwargs": {"enable_thinking": False}}
        if args.disable_thinking else None
    )
    room = client.beta.chat.completions.parse(
        model=args.model,
        messages=[{
            "role": "user",
            "content": "Candidate rooms are [1, 2]. A bed is visible in room 2. Choose the bedroom.",
        }],
        response_format=RoomChoice,
        extra_body=extra_body,
    ).choices[0].message.parsed
    if room is None or room.final_answer != 2:
        raise RuntimeError(f"unexpected room response: {room}")

    verification = client.beta.chat.completions.parse(
        model=args.model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Is this image predominantly red?"},
                {"type": "image_url", "image_url": {"url": image_data_url()}},
            ],
        }],
        response_format=Verification,
        extra_body=extra_body,
    ).choices[0].message.parsed
    if verification is None or not verification.flag:
        raise RuntimeError(f"unexpected image response: {verification}")

    print({"text": room.model_dump(), "image": verification.model_dump()})


if __name__ == "__main__":
    main()
