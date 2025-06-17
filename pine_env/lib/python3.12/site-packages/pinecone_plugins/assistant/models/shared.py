from dataclasses import dataclass

from pinecone_plugins.assistant.data.core.client.model.message_model import MessageModel as OpenAPIMessage
from pinecone_plugins.assistant.data.core.client.model.usage_model import UsageModel as OpenAPIUsageModel
from pinecone_plugins.assistant.evaluation.core.client.model.token_counts import TokenCounts as OpenAPITokenCounts
from pinecone_plugins.assistant.models.core.dataclass import BaseDataclass


@dataclass
class Message(BaseDataclass):
    content: str
    role: str = "user"

    @classmethod
    def from_openapi(cls, message_model: OpenAPIMessage):
        return cls(
            role=message_model.role,
            content=message_model.content
        )

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            role=d.get("role", "user"),
            content=d.get("content")
        )


@dataclass
class Usage(BaseDataclass):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            prompt_tokens=d.get("prompt_tokens"),
            completion_tokens=d.get("completion_tokens"),
            total_tokens=d.get("total_tokens")
        )

    @classmethod
    def from_openapi(cls, usage_model: OpenAPIUsageModel):
        return cls(
            prompt_tokens=usage_model.prompt_tokens,
            completion_tokens=usage_model.completion_tokens,
            total_tokens=usage_model.total_tokens
        )


@dataclass
class TokenCounts(BaseDataclass):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def from_openapi(cls, token_counts: OpenAPITokenCounts):
        return cls(
            prompt_tokens=token_counts.prompt_tokens,
            completion_tokens=token_counts.completion_tokens,
            total_tokens=token_counts.total_tokens
        )
