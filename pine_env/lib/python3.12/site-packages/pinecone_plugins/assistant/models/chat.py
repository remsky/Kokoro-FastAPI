from dataclasses import dataclass
from typing import List, Optional

from pinecone_plugins.assistant.data.core.client.model.chat_model import ChatModel as OpenAPIChatModel
from pinecone_plugins.assistant.data.core.client.model.citation_model import CitationModel
from pinecone_plugins.assistant.data.core.client.model.reference_model import ReferenceModel
from pinecone_plugins.assistant.models.core.dataclass import BaseDataclass
from pinecone_plugins.assistant.models.file_model import FileModel
from pinecone_plugins.assistant.models.shared import Message, Usage


@dataclass
class Highlight(BaseDataclass):
    type: str
    content: str

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            type=d.get("type"),
            content=d.get("content")
        )

    @classmethod
    def from_openapi(cls, highlight_model: dict):
        return cls(
            type=highlight_model.get("type"),
            content=highlight_model.get("content")
        )

@dataclass
class Reference(BaseDataclass):
    file: FileModel
    pages: List[int]
    highlight: Optional[Highlight]

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            pages=d.get("pages"),
            file=FileModel.from_dict(d.get("file")),
            highlight=Highlight.from_dict(d.get("highlight")) if d.get("highlight") else None
        )

    @classmethod
    def from_openapi(cls, reference_model: ReferenceModel):
        return cls(
            pages=reference_model.pages,
            file=FileModel.from_openapi(reference_model.file),
            highlight=Highlight.from_openapi(reference_model.highlight) if reference_model.highlight else None
        )


@dataclass
class Citation(BaseDataclass):
    position: int
    references: List[Reference]

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            position=d.get("position"),
            references=[Reference.from_dict(reference_data) for reference_data in d.get("references", [])]
        )

    @classmethod
    def from_openapi(cls, citation_model: CitationModel):
        return cls(
            position=citation_model.position,
            references=[Reference.from_openapi(reference_data) for reference_data in citation_model.references]
        )


@dataclass
class ChatResponse(BaseDataclass):
    id: str
    model: str
    usage: Usage
    message: Message
    finish_reason: str
    citations: List[Citation]

    @classmethod
    def from_openapi(cls, chat_model: OpenAPIChatModel):
        return cls(
            id=chat_model.id,
            model=chat_model.model,
            usage=Usage.from_openapi(chat_model.usage),
            message=Message.from_openapi(chat_model.message),
            finish_reason=chat_model.finish_reason,
            citations=[Citation.from_openapi(citation_data) for citation_data in chat_model.citations]
        )


class BaseStreamChatResponseChunk(BaseDataclass):
    pass


@dataclass
class StreamChatResponseMessageStart(BaseStreamChatResponseChunk):
    type: str
    model: str
    role: str

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            type=d.get("type"),
            model=d.get("model"),
            role=d.get("role")
        )


@dataclass
class MessageDelta(BaseDataclass):
    content: str

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            content=d.get("content")
        )


@dataclass
class StreamChatResponseContentDelta(BaseStreamChatResponseChunk):
    id: str
    type: str
    model: str
    delta: MessageDelta

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            id=d.get("id"),
            type=d.get("type"),
            model=d.get("model"),
            delta=MessageDelta.from_dict(d.get("delta"))
        )


@dataclass
class StreamChatResponseCitation(BaseStreamChatResponseChunk):
    type: str
    id: str
    model: str
    citation: Citation

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            type=d.get("type"),
            id=d.get("id"),
            model=d.get("model"),
            citation=Citation.from_dict(d.get("citation"))
        )


@dataclass
class StreamChatResponseMessageEnd(BaseStreamChatResponseChunk):
    type: str
    model: str
    id: str
    usage: Usage

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            type=d.get("type"),
            model=d.get("model"),
            id=d.get("id"),
            usage=Usage.from_dict(d.get("usage"))
        )

@dataclass
class ContextOptions(BaseDataclass):
    top_k: Optional[int] = None
    snippet_size: Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            top_k=d.get("top_k"),
            snippet_size=d.get("snippet_size")
        )