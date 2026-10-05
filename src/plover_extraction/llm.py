"""Small OpenAI-compatible chat client used by the extraction pipeline."""

import json
import os
from typing import Protocol

from openai import OpenAIError, OpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import LLMConfig
import httpx

class LLMClient(Protocol):
    async def generate_json(self, system_prompt: str, user_prompt: str) -> object:
        """Return the decoded JSON value produced by an LLM."""


class OpenAICompatibleClient:
    """Call an OpenAI-compatible endpoint using the OpenAI SDK."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

        print(f"Initializing OpenAI-compatible client for model {config.model} at {config.api_base}, with {config.api_key_env}...")

        api_key =config.api_key_env
        
        if not api_key:
            raise RuntimeError(
                f"Missing API key environment variable: {config.api_key_env}"
            )
        # self.client = OpenAI(
        #     api_key=api_key,
        #     base_url=config.api_base,
        #     timeout=config.timeout_seconds,
        #     max_retries=config.max_retries,
        #     http_client=httpx.Client(
        #             verify=False,
        #             timeout=300.0,
        #         )
        #     # extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        # )

        self.http_client = httpx.AsyncClient(
            verify=False,
            timeout=config.timeout_seconds,
        )

        self.client = ChatOpenAI(
            model=config.model,
            base_url=config.api_base,
            api_key=api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
            temperature=config.temperature,
            max_tokens=4096,
            streaming=True,
            stream_usage=False,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            },
            http_async_client=self.http_client,
        )

    async def generate_json(self, system_prompt: str, user_prompt: str) -> object:
        full_message = None

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{user_prompt}\n/no_think"),
        ]

        try:
            print("Sending request to LLM...")
            print("system_prompt:", system_prompt)
            print("user_prompt:", user_prompt)
            print("LLM content:", flush=True)

            stream = self.client.astream(messages)

            async for chunk in stream:
                if full_message is None:
                    full_message = chunk
                else:
                    full_message = full_message + chunk

                if isinstance(chunk.content, str):
                    token_text = chunk.content
                else:
                    token_text = "".join(
                        block.get("text", "")
                        if isinstance(block, dict)
                        and block.get("type") == "text"
                        else block if isinstance(block, str) else ""
                        for block in chunk.content
                    )

                if token_text:
                    print(token_text, end="", flush=True)

        except OpenAIError as error:
            raise RuntimeError(
                f"LLM request failed: {error}"
            ) from error
        finally:
            print()

        if full_message is None:
            raise RuntimeError("LLM returned no response.")

        refusal = full_message.additional_kwargs.get("refusal")
        if refusal:
            raise RuntimeError(
                f"LLM refused the request: {refusal}"
            )

        finish_reason = full_message.response_metadata.get(
            "finish_reason"
        )
        if finish_reason != "stop":
            raise RuntimeError(
                f"LLM response incomplete: finish_reason={finish_reason}"
            )

        if isinstance(full_message.content, str):
            content = full_message.content.strip()
        else:
            content = "".join(
                block.get("text", "")
                if isinstance(block, dict)
                and block.get("type") == "text"
                else block if isinstance(block, str) else ""
                for block in full_message.content
            ).strip()

        if not content:
            raise RuntimeError("LLM returned empty content.")

        print("Received complete response from LLM.")

        try:
            return json.loads(content)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"LLM returned invalid JSON: {error}"
            ) from error

    async def close(self) -> None:
        await self.http_client.aclose()

    # def generate_json(self, system_prompt: str, user_prompt: str) -> object:
    #     content_parts: list[str] = []
    #     refusal_parts: list[str] = []
    #     finish_reason = None

    #     try:
    #         print("Sending request to LLM...")
    #         print("system_prompt:", system_prompt)
    #         print("user_prompt:", user_prompt)

    #         with self.client.chat.completions.create(
    #             model=self.config.model,
    #             temperature=self.config.temperature,
    #             # response_format={"type": "json_object"},
    #             messages=[
    #                 {"role": "system", "content": system_prompt},
    #                 {
    #                     "role": "user",
    #                     "content": f"{user_prompt}\n/no_think",
    #                 },
    #             ],
    #             extra_body={
    #                 "chat_template_kwargs": {
    #                     "enable_thinking": False,
    #                 },
    #             },
    #             stream=True,
    #         ) as stream:
    #             print("LLM content:", flush=True)

    #             for chunk in stream:
    #                 if not chunk.choices:
    #                     continue

    #                 choice = chunk.choices[0]
    #                 delta = choice.delta

    #                 if delta.content:
    #                     print(delta.content, end="", flush=True)
    #                     content_parts.append(delta.content)

    #                 if delta.refusal:
    #                     refusal_parts.append(delta.refusal)

    #                 if choice.finish_reason is not None:
    #                     finish_reason = choice.finish_reason

    #     except OpenAIError as error:
    #         raise RuntimeError(f"LLM request failed: {error}") from error
    #     finally:
    #         print()

    #     content = "".join(content_parts).strip()
    #     refusal = "".join(refusal_parts).strip()

    #     if refusal:
    #         raise RuntimeError(f"LLM refused the request: {refusal}")

    #     if finish_reason != "stop":
    #         raise RuntimeError(
    #             f"LLM response incomplete: finish_reason={finish_reason}"
    #         )

    #     if not content:
    #         raise RuntimeError("LLM returned empty content.")

    #     print("Received complete response from LLM.")

    #     try:
    #         return json.loads(content)
    #     except json.JSONDecodeError as error:
    #         raise RuntimeError(
    #             f"LLM returned invalid JSON: {error}"
    #         ) from error
    
    # def close(self) -> None:
    #     self.client.close()