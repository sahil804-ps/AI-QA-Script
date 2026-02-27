"""
ai_clients.py — Multi-AI client wrappers for GPT, Gemini, and Claude

Each client has a .get_response(prompt) method.
Missing API keys are handled gracefully — client is marked unavailable.
"""

import os
import concurrent.futures
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# ═══════════════════════════════════════════════════════════════
# Response container
# ═══════════════════════════════════════════════════════════════

@dataclass
class AIResponse:
    ai_name: str          # e.g. "GPT-4o-mini"
    model: str            # e.g. "gpt-4o-mini"
    response: Optional[str]
    success: bool
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# Base Client
# ═══════════════════════════════════════════════════════════════

class BaseAIClient:
    ai_name: str = "Base"
    model: str = "base"

    def is_available(self) -> bool:
        raise NotImplementedError

    def get_response(self, prompt: str) -> AIResponse:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════
# OpenAI GPT Client
# ═══════════════════════════════════════════════════════════════

class OpenAIClient(BaseAIClient):
    ai_name = "ChatGPT"
    model = "gpt-4o-mini"

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()

    def is_available(self) -> bool:
        return bool(self.api_key)

    def get_response(self, prompt: str) -> AIResponse:
        if not self.is_available():
            return AIResponse(
                ai_name=self.ai_name, model=self.model,
                response=None, success=False,
                error="OPENAI_API_KEY not set in .env"
            )
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            result = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7,
            )
            text = result.choices[0].message.content.strip()
            return AIResponse(ai_name=self.ai_name, model=self.model,
                              response=text, success=True)
        except Exception as e:
            return AIResponse(ai_name=self.ai_name, model=self.model,
                              response=None, success=False, error=str(e))


# ═══════════════════════════════════════════════════════════════
# Google Gemini Client
# ═══════════════════════════════════════════════════════════════

class GeminiClient(BaseAIClient):
    ai_name = "Gemini"
    model = "gemini-1.5-flash"

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()

    def is_available(self) -> bool:
        return bool(self.api_key)

    def get_response(self, prompt: str) -> AIResponse:
        if not self.is_available():
            return AIResponse(
                ai_name=self.ai_name, model=self.model,
                response=None, success=False,
                error="GEMINI_API_KEY not set in .env"
            )
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            result = model.generate_content(prompt)
            text = result.text.strip()
            return AIResponse(ai_name=self.ai_name, model=self.model,
                              response=text, success=True)
        except Exception as e:
            return AIResponse(ai_name=self.ai_name, model=self.model,
                              response=None, success=False, error=str(e))


# ═══════════════════════════════════════════════════════════════
# Anthropic Claude Client
# ═══════════════════════════════════════════════════════════════

class AnthropicClient(BaseAIClient):
    ai_name = "Claude"
    model = "claude-3-haiku-20240307"

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    def is_available(self) -> bool:
        return bool(self.api_key)

    def get_response(self, prompt: str) -> AIResponse:
        if not self.is_available():
            return AIResponse(
                ai_name=self.ai_name, model=self.model,
                response=None, success=False,
                error="ANTHROPIC_API_KEY not set in .env"
            )
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = message.content[0].text.strip()
            return AIResponse(ai_name=self.ai_name, model=self.model,
                              response=text, success=True)
        except Exception as e:
            return AIResponse(ai_name=self.ai_name, model=self.model,
                              response=None, success=False, error=str(e))


# ═══════════════════════════════════════════════════════════════
# Multi-AI Runner — calls all AIs in parallel
# ═══════════════════════════════════════════════════════════════

ALL_CLIENTS = [OpenAIClient, GeminiClient, AnthropicClient]


def fetch_all_responses(prompt: str, selected: Optional[list] = None) -> list[AIResponse]:
    """
    Calls all available AI clients in parallel.
    selected: optional list of names e.g. ["gpt", "gemini"]
    Returns list of AIResponse (including failed ones).
    """
    name_map = {
        "gpt": OpenAIClient,
        "chatgpt": OpenAIClient,
        "openai": OpenAIClient,
        "gemini": GeminiClient,
        "google": GeminiClient,
        "claude": AnthropicClient,
        "anthropic": AnthropicClient,
    }

    if selected:
        clients_to_use = []
        for s in selected:
            cls = name_map.get(s.lower())
            if cls and cls not in [type(c) for c in clients_to_use]:
                clients_to_use.append(cls())
    else:
        clients_to_use = [cls() for cls in ALL_CLIENTS]

    def call_one(client: BaseAIClient) -> AIResponse:
        return client.get_response(prompt)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(clients_to_use)) as executor:
        futures = {executor.submit(call_one, c): c for c in clients_to_use}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Sort by ai_name for consistent display order
    results.sort(key=lambda r: r.ai_name)
    return results
