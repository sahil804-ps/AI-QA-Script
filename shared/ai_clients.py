"""
ai_clients.py — Multi-AI client wrappers for GPT, Gemini, and Claude

Each client has a .get_response(prompt) method.
Missing API keys are handled gracefully — client is marked unavailable.
"""

import os
import time
import threading
import concurrent.futures
from dataclasses import dataclass
from typing import Optional, List

from dotenv import load_dotenv
from colorama import Fore, Style

load_dotenv()

# Utility for colors
def _col(text, color):
    return f"{color}{text}{Style.RESET_ALL}"

# ═══════════════════════════════════════════════════════════════
# Spinner for loading animation
# ═══════════════════════════════════════════════════════════════

class Spinner:
    CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str):
        self.message = message
        self._running = False
        self._thread = None

    def _spin(self):
        i = 0
        while self._running:
            char = self.CHARS[i % len(self.CHARS)]
            print(f"\r  {Fore.CYAN}{char}{Style.RESET_ALL}  {self.message}", end="", flush=True)
            time.sleep(0.08)
            i += 1

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self, success: bool = True, label: str = ""):
        self._running = False
        if self._thread:
            self._thread.join()
        icon = f"{Fore.GREEN}✅" if success else f"{Fore.RED}❌"
        suffix = f"  {Fore.WHITE}{label}" if label else ""
        print(f"\r  {icon}  {self.message}{suffix}{Style.RESET_ALL}")



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
# OpenRouter Client
# ═══════════════════════════════════════════════════════════════

class OpenRouterClient(BaseAIClient):
    """Client for OpenRouter (provides access to Llama, Gemini, etc. via one key)"""
    def __init__(self, model="google/gemini-2.0-flash-001"):
        self.ai_name = "OpenRouter"
        self.model = model
        self.api_key = os.getenv("OPENROUTER_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def get_response(self, prompt: str) -> AIResponse:
        if not self.api_key:
            return AIResponse(self.ai_name, self.model, None, False, "API key not set")
        
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )
            completion = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                extra_headers={
                    "HTTP-Referer": "https://github.com/AI-QA-Toolkit",
                    "X-Title": "AI-QA-Toolkit",
                }
            )
            text = completion.choices[0].message.content
            return AIResponse(self.ai_name, self.model, text.strip(), success=True)
        except Exception as e:
            return AIResponse(self.ai_name, self.model, None, False, str(e))


class LlamaClient(OpenRouterClient):
    """Llama 3.3 70B via OpenRouter"""
    def __init__(self):
        super().__init__(model="meta-llama/llama-3.3-70b-instruct")
        self.ai_name = "Llama-3"


class DeepSeekClient(OpenRouterClient):
    """DeepSeek V3 via OpenRouter"""
    def __init__(self):
        super().__init__(model="deepseek/deepseek-chat")
        self.ai_name = "DeepSeek"


# ═══════════════════════════════════════════════════════════════
# Multi-AI Runner — calls all AIs in parallel
# ═══════════════════════════════════════════════════════════════
# Default export for all clients
ALL_CLIENTS = [OpenAIClient, GeminiClient, AnthropicClient, OpenRouterClient, LlamaClient, DeepSeekClient]


def fetch_all_responses(prompt: str, selected: Optional[list] = None) -> list[AIResponse]:
    """
    Calls all available AI clients in parallel without UI.
    """
    name_map = {
        "gpt": OpenAIClient, "chatgpt": OpenAIClient, "openai": OpenAIClient,
        "gemini": GeminiClient, "google": GeminiClient,
        "claude": AnthropicClient, "anthropic": AnthropicClient,
        "openrouter": OpenRouterClient, "or": OpenRouterClient,
        "llama": LlamaClient, "meta": LlamaClient,
        "deepseek": DeepSeekClient, "ds": DeepSeekClient,
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

    results.sort(key=lambda r: r.ai_name)
    return results


def fetch_with_spinners(prompt: str, selected: Optional[list] = None) -> List[AIResponse]:
    """Fetch responses from all AIs, showing a spinner per AI."""
    name_map = {
        "gpt": OpenAIClient, "chatgpt": OpenAIClient, "openai": OpenAIClient,
        "gemini": GeminiClient, "google": GeminiClient,
        "claude": AnthropicClient, "anthropic": AnthropicClient,
        "openrouter": OpenRouterClient, "or": OpenRouterClient,
        "llama": LlamaClient, "meta": LlamaClient,
        "deepseek": DeepSeekClient, "ds": DeepSeekClient,
    }

    if selected:
        seen = set()
        clients = []
        for s in selected:
            cls = name_map.get(s.lower())
            if cls and cls not in seen:
                clients.append(cls())
                seen.add(cls)
    else:
        clients = [cls() for cls in ALL_CLIENTS]

    print()
    print(_col("  🚀  Fetching responses from AIs...", Fore.CYAN))
    print()

    results = []
    result_lock = threading.Lock()
    spinners: dict = {}

    # Start all spinners
    for client in clients:
        sp = Spinner(f"Calling {client.ai_name} ({client.model})...")
        spinners[client.ai_name] = sp
        sp.start()

    def call_one(client):
        if not client.is_available():
            resp = AIResponse(
                ai_name=client.ai_name, model=client.model,
                response=None, success=False,
                error="API key not set"
            )
        else:
            resp = client.get_response(prompt)

        sp = spinners[client.ai_name]
        if resp.success:
            word_count = len(resp.response.split())
            sp.stop(success=True, label=f"({word_count} words)")
        else:
            sp.stop(success=False, label=f"({resp.error[:50]})")

        with result_lock:
            results.append(resp)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(clients)) as executor:
        futures = [executor.submit(call_one, c) for c in clients]
        concurrent.futures.wait(futures)

    results.sort(key=lambda r: r.ai_name)
    return results
