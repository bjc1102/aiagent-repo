from __future__ import annotations

import base64
from pathlib import Path
from time import perf_counter
from typing import Any

import os

from dotenv import load_dotenv
from openai import NotFoundError, OpenAI

from prompts import build_system_prompt
from schemas.copayment_response import CopaymentResponse

load_dotenv()

_IMAGE_EXTRACT_SYSTEM = """당신은 의료급여·건강보험 안내 자료 이미지에서 텍스트와 표를 구조화해 옮기는 역할만 합니다.
추측으로 숫자를 바꾸지 말고, 이미지에 보이는 구분·비율·주석을 그대로 반영하세요.
출력은 반드시 마크다운(표, 글머리 기호)으로만 작성하고, JSON이나 다른 형식은 사용하지 마세요."""

_IMAGE_EXTRACT_USER_TEXT = """이 이미지에 나온 의료급여 본인부담률(또는 본인부담금) 관련 표·설명을 모두 추출하세요.

요구사항:
- 섹션 번호(예: 04, 05…)와 표 제목이 있으면 유지합니다.
- 1종/2종, 연령, 입원/외래, 의료기관 차수, 질환/시술 구분 등 조건 열을 빠뜨리지 않습니다.
- 각주·예외 문구가 있으면 같은 섹션 아래에 적습니다.
- 이미지에 없는 규칙은 만들어 내지 마세요."""

_DEFAULT_MODEL = "gemini-2.5-flash"
_MODEL_ALIASES = {
    "gemini-3.0-flash-preview": _DEFAULT_MODEL,
}
_FALLBACK_MODELS = (_DEFAULT_MODEL, "gemini-flash-latest")


class GeminiService:
    """Gemini OpenAI 호환 엔드포인트로 구조화 JSON 응답을 받습니다 (week-1 V2 패턴)."""

    def __init__(
        self,
        *,
        prompt_profile: str | None = None,
        system_prompt_override: str | None = None,
        reference_text: str | None = None,
    ) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        # 기본값은 현재 OpenAI 호환 엔드포인트에서 안정적으로 동작하는 Flash 계열로 둡니다.
        self.model_name = self._normalize_model_name(os.getenv("GEMINI_MODEL", _DEFAULT_MODEL))
        # reasoning_effort: 일부 Gemini(특히 추론/사고 토큰을 쓰는 계열)에서
        # 내부 추론(Thinking) 깊이를 조절하는 옵션입니다. OpenAI 호환 호출에서는
        # `extra_body={"reasoning_effort": "low"|"medium"|"high"}` 로 전달됩니다.
        # 지원하지 않는 모델이면 무시되거나 오류 날 수 있어, 끄려면 GEMINI_REASONING_EFFORT=off
        _effort = os.getenv("GEMINI_REASONING_EFFORT", "low").strip()
        self._reasoning_effort_disabled = _effort.lower() in ("", "off", "none", "false", "0")
        self.reasoning_effort = "" if self._reasoning_effort_disabled else _effort

        if system_prompt_override is not None:
            self.prompt_profile = prompt_profile or "custom"
            self.system_prompt = system_prompt_override
        else:
            profile = prompt_profile or os.getenv("PROMPT_PROFILE", "zero_shot")
            self.prompt_profile = profile
            self.system_prompt = build_system_prompt(profile, reference_text=reference_text)

        self.generation_defaults = {
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 1024,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "seed": None,
        }

        self.response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "copayment_response",
                "schema": CopaymentResponse.model_json_schema(),
            },
        }

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY가 환경 변수 또는 .env에 없습니다.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        normalized = model_name.strip()
        if normalized.startswith("models/"):
            normalized = normalized.split("/", 1)[1]
        return _MODEL_ALIASES.get(normalized, normalized)

    def _create_chat_completion(self, **create_kwargs: Any) -> Any:
        requested_model = self._normalize_model_name(str(create_kwargs["model"]))
        candidates = [requested_model]
        for fallback in _FALLBACK_MODELS:
            if fallback not in candidates:
                candidates.append(fallback)

        last_error: NotFoundError | None = None
        for model_name in candidates:
            try:
                create_kwargs["model"] = model_name
                response = self.client.chat.completions.create(**create_kwargs)
                self.model_name = model_name
                return response
            except NotFoundError as exc:
                last_error = exc
                message = str(exc)
                if "is not found" not in message and "no longer available" not in message:
                    raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("모델 호출에 실패했습니다.")

    def _resolve_generation_options(
        self,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        opts = dict[str, Any](self.generation_defaults)
        if temperature is not None:
            opts["temperature"] = temperature
        if top_p is not None:
            opts["top_p"] = top_p
        if max_tokens is not None:
            opts["max_tokens"] = max_tokens
        if presence_penalty is not None:
            opts["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            opts["frequency_penalty"] = frequency_penalty
        if seed is not None:
            opts["seed"] = seed
        return opts

    @staticmethod
    def _extract_token_usage(response: Any) -> dict[str, int | None]:
        usage = getattr(response, "usage", None)
        usage_data: dict[str, Any] = {}
        if usage is not None:
            if hasattr(usage, "model_dump"):
                usage_data = usage.model_dump(exclude_none=True)
            elif isinstance(usage, dict):
                usage_data = usage

        prompt_tokens = usage_data.get("prompt_tokens", usage_data.get("input_tokens"))
        completion_tokens = usage_data.get("completion_tokens", usage_data.get("output_tokens"))
        total_tokens = usage_data.get("total_tokens")
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _image_path_to_data_url(image_path: Path) -> str:
        raw = image_path.read_bytes()
        b64 = base64.standard_b64encode(raw).decode("ascii")
        ext = image_path.suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(ext, "image/png")
        return f"data:{mime};base64,{b64}"

    def extract_reference_from_image(
        self,
        image_path: Path,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
    ) -> tuple[str, dict[str, int | float | None]]:
        """이미지에서 본인부담률 참조 텍스트를 추출합니다 (비전 1회 호출, 일반 텍스트 응답)."""
        path = image_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"이미지 파일이 없습니다: {path}")

        extract_max = (
            max_tokens
            if max_tokens is not None
            else int(os.getenv("GEN_IMAGE_EXTRACT_MAX_TOKENS", "8192"))
        )
        gen = self._resolve_generation_options(
            temperature=0.1 if temperature is None else temperature,
            top_p=top_p,
            max_tokens=extract_max,
        )

        data_url = self._image_path_to_data_url(path)
        started = perf_counter()
        create_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": _IMAGE_EXTRACT_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _IMAGE_EXTRACT_USER_TEXT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "temperature": gen["temperature"],
            "top_p": gen["top_p"],
            "max_tokens": gen["max_tokens"],
        }
        if self.reasoning_effort:
            create_kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        response = self._create_chat_completion(**create_kwargs)
        elapsed_ms = round((perf_counter() - started) * 1000, 2)

        text = (response.choices[0].message.content or "").strip()
        meta = self._extract_token_usage(response)
        meta["elapsed_ms"] = elapsed_ms
        meta["image_path"] = str(path)
        return text, meta

    def get_config(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "base_url": self.base_url,
            "prompt_profile": self.prompt_profile,
            "generation_defaults": self.generation_defaults,
            "system_prompt": self.system_prompt,
            "response_format": self.response_format,
            "reasoning_effort": self.reasoning_effort or None,
            "reasoning_effort_enabled": bool(self.reasoning_effort),
        }

    def answer_with_usage(
        self,
        question: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
    ) -> tuple[CopaymentResponse, dict[str, int | float | None]]:
        """단일 질문에 대해 CopaymentResponse와 usage 메타데이터를 반환합니다."""
        sys_content = system_prompt if system_prompt is not None else self.system_prompt
        gen = self._resolve_generation_options(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
        )

        started = perf_counter()
        create_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": question},
            ],
            "temperature": gen["temperature"],
            "top_p": gen["top_p"],
            "max_tokens": gen["max_tokens"],
            "presence_penalty": gen["presence_penalty"],
            "response_format": self.response_format,
        }
        if self.reasoning_effort:
            create_kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        response = self._create_chat_completion(**create_kwargs)
        elapsed_ms = round((perf_counter() - started) * 1000, 2)

        raw = response.choices[0].message.content
        if not raw:
            raise ValueError("모델 응답이 비어 있습니다.")
        parsed = CopaymentResponse.model_validate_json(raw)
        meta = self._extract_token_usage(response)
        meta["elapsed_ms"] = elapsed_ms
        return parsed, meta
