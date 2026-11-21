"""TokenHub Chat Completions transport and benchmark-compatible interfaces."""

import base64
import math
import os
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from threading import local

from openai import APIConnectionError, APIResponseValidationError, APIStatusError, APITimeoutError, OpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain_core.runnables import RunnableLambda


DEFAULT_BASE_URL = "https://tokenhub-intl.tencentcloudmaas.com/v1"
RESPONSE_POLICY = "upstream-text-v1"


class TokenHubError(RuntimeError):
    """Sanitized adapter error; never includes the SDK payload or credentials."""

    def __init__(self, code, status_code=None):
        self.code = code
        self.status_code = status_code
        suffix = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"TokenHub {code}{suffix}")


class TokenHubPermanentError(TokenHubError):
    """Invalid configuration, request, or final response; must not be retried."""


class TokenHubRetryExhaustedError(TokenHubError):
    """A transient failure has already exhausted the adapter's retry budget."""


def is_permanent_error(error):
    return isinstance(error, TokenHubPermanentError)


def get_tokenhub_model_id():
    model_id = os.getenv("TOKENHUB_MODEL_ID")
    if not model_id or not model_id.strip():
        raise TokenHubPermanentError("missing_model_id")
    return model_id


class TokenHubAdapter:
    def __init__(self, *, model_id=None, max_retries=3, retry_delay=1.0,
                 retry_delay_cap=30.0, client=None):
        self.model_id = get_tokenhub_model_id() if model_id is None else model_id
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise TokenHubPermanentError("missing_model_id")
        if not isinstance(max_retries, int) or max_retries < 0:
            raise TokenHubPermanentError("invalid_retry_config")
        try:
            self.retry_delay = float(retry_delay)
            self.retry_delay_cap = float(retry_delay_cap)
            self.timeout = float(os.getenv("TOKENHUB_TIMEOUT", "180"))
        except (TypeError, ValueError):
            raise TokenHubPermanentError("invalid_numeric_config") from None
        if (not math.isfinite(self.retry_delay) or self.retry_delay < 0
                or not math.isfinite(self.retry_delay_cap) or self.retry_delay_cap <= 0
                or not math.isfinite(self.timeout) or self.timeout <= 0):
            raise TokenHubPermanentError("invalid_numeric_config")
        self.max_retries = max_retries
        self.thinking = os.getenv("TOKENHUB_THINKING")
        if self.thinking is not None and self.thinking not in ("enabled", "disabled"):
            raise TokenHubPermanentError("invalid_thinking")
        self.reasoning_effort = os.getenv("TOKENHUB_REASONING_EFFORT")
        if client is None:
            api_key = os.getenv("TOKENHUB_API_KEY") or os.getenv("TENCENT_TOKENHUB_API_KEY")
            if not api_key:
                raise TokenHubPermanentError("missing_api_key")
            try:
                client = OpenAI(
                    api_key=api_key,
                    base_url=os.getenv("TOKENHUB_BASE_URL") or DEFAULT_BASE_URL,
                    timeout=self.timeout,
                    max_retries=0,
                )
            except Exception:
                raise TokenHubPermanentError("invalid_client_config") from None
        self.client = client

    @staticmethod
    def _classify_error(error):
        if isinstance(error, (APITimeoutError, TimeoutError)):
            return "timeout", None, True
        if isinstance(error, (APIConnectionError, ConnectionError)):
            return "connection_error", None, True
        if isinstance(error, APIStatusError):
            status = error.status_code
            model_errors = {"modelnotfound", "modelunavailable", "modelnotavailable",
                            "modelnotsupported", "unsupportedmodel", "invalidmodel"}
            if any(isinstance(value, str) and value.lower().replace("_", "") in model_errors
                   for value in (getattr(error, "code", None), getattr(error, "type", None))):
                return "model_unavailable", status, False
            return "http_error", status, status in (408, 429) or 500 <= status < 600
        if isinstance(error, APIResponseValidationError):
            return "response_schema", None, False
        return "request_error", None, False

    def _retry_after(self, error):
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {})
        value = headers.get("retry-after", headers.get("Retry-After"))
        if value is None:
            return 0.0
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                seconds = (date - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                return 0.0
        if not math.isfinite(seconds):
            return 0.0
        return min(self.retry_delay_cap, max(0.0, seconds))

    @staticmethod
    def _final_content(response):
        choices = getattr(response, "choices", None)
        if not isinstance(choices, (list, tuple)) or not choices:
            raise TokenHubPermanentError("response_schema")
        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason in ("content_filter", "tool_calls"):
            code = {"content_filter": "content_filtered",
                    "tool_calls": "tool_calls"}[finish_reason]
            raise TokenHubPermanentError(code)
        if finish_reason not in ("stop", "length"):
            raise TokenHubPermanentError("response_schema")
        message = getattr(choice, "message", None)
        if getattr(message, "tool_calls", None):
            raise TokenHubPermanentError("tool_calls")
        content = getattr(message, "content", None)
        if content is None and getattr(message, "role", None) == "assistant":
            return ""
        if not isinstance(content, str):
            raise TokenHubPermanentError("missing_final_content")
        return content

    def complete(self, messages, *, model_id=None, max_tokens=None, temperature=None):
        selected_model = self.model_id if model_id is None else model_id
        if not isinstance(selected_model, str) or not selected_model.strip():
            raise TokenHubPermanentError("missing_model_id")
        request = {"model": selected_model, "messages": messages}
        kimi_completion = selected_model in {
            "kimi-k3", "kimi-k3.1-pro", "kimi-k3.1-pro-highspeed",
        }
        kimi_fixed_sampling = kimi_completion or selected_model in {
            "kimi-k2.8-preview", "kimi-k2.7-code", "kimi-k2.7-code-highspeed",
        }
        minimax_split = selected_model in {"minimax-m3", "minimax-m3.1-flash"}
        if max_tokens is not None:
            request["max_completion_tokens" if kimi_completion else "max_tokens"] = max_tokens
        # These models impose temperature=1.0; omit unsupported overrides.
        # This API constraint must be disclosed in comparisons, not normalized away.
        if temperature is not None and not (kimi_fixed_sampling or minimax_split):
            request["temperature"] = temperature
        extra_body = {"reasoning_split": True} if minimax_split else {}
        if self.thinking is not None:
            extra_body["thinking"] = {"type": self.thinking}
        if extra_body:
            request["extra_body"] = extra_body
        if self.reasoning_effort is not None:
            request["reasoning_effort"] = self.reasoning_effort

        delay = min(self.retry_delay, self.retry_delay_cap)
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**request)
            except Exception as error:
                code, status, retryable = self._classify_error(error)
                if not retryable:
                    raise TokenHubPermanentError(code, status) from None
                if attempt == self.max_retries:
                    raise TokenHubRetryExhaustedError(code, status) from None
                time.sleep(min(self.retry_delay_cap, max(delay, self._retry_after(error))))
                delay = min(self.retry_delay_cap, delay * 2)
                continue
            return self._final_content(response)


def _prompt_messages(prompt, system_prompt):
    if isinstance(prompt, (StringPromptValue, str)):
        text = prompt.text if isinstance(prompt, StringPromptValue) else prompt
        # The upstream OpenAI-compatible basic template sends only the user text.
        # A system message must come from the selected chat template, not the adapter.
        return [{"role": "user", "content": text}]
    if not isinstance(prompt, ChatPromptValue):
        raise TokenHubPermanentError("unsupported_prompt_input")
    roles = {"system": "system", "human": "user", "ai": "assistant",
             "tool": "tool", "function": "function"}
    messages = []
    for message in prompt.to_messages():
        role = message.role if message.type == "chat" else roles.get(message.type)
        if role is None:
            raise TokenHubPermanentError("unsupported_message_role")
        converted = {"role": role, "content": message.content}
        if message.name is not None:
            converted["name"] = message.name
        if message.type == "tool":
            converted["tool_call_id"] = message.tool_call_id
        if message.additional_kwargs:
            converted.update(message.additional_kwargs)
        if message.type == "ai" and message.tool_calls:
            raise TokenHubPermanentError("unsupported_prompt_tool_calls")
        messages.append(converted)
    return messages


def get_tokenhub_llm(*, temperature=0.01, max_tokens=256, max_retries=3,
                     wait_time=1.0, system_prompt=None):
    adapter = TokenHubAdapter(max_retries=max_retries, retry_delay=wait_time)

    def invoke(prompt):
        content = adapter.complete(
            _prompt_messages(prompt, system_prompt),
            max_tokens=max_tokens, temperature=temperature,
        )
        return AIMessage(content=content)

    return RunnableLambda(invoke), adapter.model_id


class TokenHubConverseClient:
    def __init__(self, *, max_retries=3, retry_delay=1.0, retry_delay_cap=30.0,
                 adapter=None, model_id=None):
        self.adapter = adapter if adapter is not None else TokenHubAdapter(
            model_id=model_id, max_retries=max_retries,
            retry_delay=retry_delay, retry_delay_cap=retry_delay_cap,
        )
        self._state = local()

    @property
    def last_error(self):
        """The sanitized terminal error for this thread's most recent call."""
        return getattr(self._state, "last_error", None)

    @staticmethod
    def _content_blocks(blocks, *, system=False):
        content = []
        for block in blocks:
            if not isinstance(block, dict) or len(block) != 1:
                raise TokenHubPermanentError("unsupported_content_block")
            if "text" in block and isinstance(block["text"], str):
                content.append({"type": "text", "text": block["text"]})
            elif "image" in block and not system:
                image = block["image"]
                try:
                    image_format = image["format"]
                    image_bytes = image["source"]["bytes"]
                except (KeyError, TypeError):
                    raise TokenHubPermanentError("invalid_image_block") from None
                if (not isinstance(image_format, str) or not image_format
                        or not isinstance(image_bytes, (bytes, bytearray, memoryview))):
                    raise TokenHubPermanentError("invalid_image_block")
                encoded = base64.b64encode(image_bytes).decode("ascii")
                content.append({"type": "image_url", "image_url": {
                    "url": f"data:image/{image_format};base64,{encoded}"}})
            else:
                raise TokenHubPermanentError("unsupported_content_block")
        return content

    def converse(self, modelId, messages, system=None, inferenceConfig=None):
        self._state.last_error = None
        try:
            converted = []
            if system:
                converted.append({"role": "system", "content": self._content_blocks(system, system=True)})
            for message in messages:
                converted.append({"role": message["role"],
                                  "content": self._content_blocks(message["content"])})
            config = inferenceConfig if inferenceConfig is not None else {}
            if not isinstance(config, dict):
                raise TokenHubPermanentError("request_schema")
            if set(config) - {"maxTokens", "temperature"}:
                raise TokenHubPermanentError("unsupported_inference_config")
            answer = self.adapter.complete(
                converted, model_id=modelId, max_tokens=config.get("maxTokens"),
                temperature=config.get("temperature"),
            )
        except TokenHubError as error:
            self._state.last_error = error
            raise
        except (KeyError, TypeError, ValueError):
            error = TokenHubPermanentError("request_schema")
            self._state.last_error = error
            raise error from None
        return {"output": {"message": {"content": [{"text": answer}]}}}
