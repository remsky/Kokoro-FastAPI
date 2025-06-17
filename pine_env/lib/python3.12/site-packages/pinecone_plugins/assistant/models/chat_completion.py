from dataclasses import dataclass
from typing import List

from pinecone_plugins.assistant.data.core.client.model.chat_completion_model import \
    ChatCompletionModel as OpenAPIChatCompletionModel
from pinecone_plugins.assistant.models.core.dataclass import BaseDataclass
from pinecone_plugins.assistant.models.shared import Message, Usage


@dataclass
class ChatCompletionChoice(BaseDataclass):
    index: int
    message: Message
    finish_reason: str


@dataclass
class ChatCompletionResponse(BaseDataclass):
    id: str
    choices: List[ChatCompletionChoice]
    model: str
    usage: Usage

    @classmethod
    def from_openapi(cls, chat_completion_model: OpenAPIChatCompletionModel):
        return cls(
            id=chat_completion_model.id,
            choices=[ChatCompletionChoice(index=choice.index,
                                          message=Message.from_openapi(choice.message),
                                          finish_reason=choice.finish_reason)
                     for choice in chat_completion_model.choices],
            model=chat_completion_model.model,
            usage=Usage.from_openapi(chat_completion_model.usage)
        )


@dataclass
class StreamingChatCompletionChoice(BaseDataclass):
    index: int
    delta: Message
    finish_reason: str

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            index=d.get("index"),
            delta=Message.from_dict(d.get("delta")),
            finish_reason=d.get("finish_reason")
        )


@dataclass
class StreamingChatCompletionChunk(BaseDataclass):
    id: str
    choices: List[StreamingChatCompletionChoice]
    model: str

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            id=d.get("id"),
            choices=[StreamingChatCompletionChoice.from_dict(choice_data) for choice_data in d.get("choices", [])],
            model=d.get("model")
        )