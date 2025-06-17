from dataclasses import dataclass
from typing import TypeVar, Union

from pinecone_plugins.assistant.data.core.client.model.context_model import (
    ContextModel as OpenAPIContextModel,
)
from pinecone_plugins.assistant.data.core.client.model.snippet_model import (
    SnippetModel as OpenAPISnippetModel,
)
from pinecone_plugins.assistant.models.core.dataclass import BaseDataclass
from pinecone_plugins.assistant.models.file_model import FileModel
from pinecone_plugins.assistant.models.shared import TokenCounts

RefType = TypeVar(
    "RefType",
    bound=Union["TextReference", "PdfReference", "MarkdownReference", "JsonReference"],
)


@dataclass
class BaseReference(BaseDataclass):
    type: str

    @classmethod
    def from_openapi(cls, value):
        raise NotImplementedError


@dataclass
class PdfReference(BaseReference):
    pages: list[int]
    file: FileModel

    @classmethod
    def from_openapi(cls, ref_dict: dict) -> "PdfReference":
        return cls(
            type=ref_dict["type"],
            pages=ref_dict["pages"],
            file=FileModel.from_dict(ref_dict["file"]),
        )


@dataclass
class DocxReference(BaseReference):
    pages: list[int]
    file: FileModel

    @classmethod
    def from_openapi(cls, ref_dict: dict) -> "PdfReference":
        return cls(
            type=ref_dict["type"],
            pages=ref_dict["pages"],
            file=FileModel.from_dict(ref_dict["file"]),
        )


@dataclass
class TextReference(BaseReference):
    file: FileModel

    @classmethod
    def from_openapi(cls, ref_dict: dict) -> "TextReference":
        return cls(type=ref_dict["type"], file=FileModel.from_dict(ref_dict["file"]))


@dataclass
class MarkdownReference(BaseReference):
    file: FileModel

    @classmethod
    def from_openapi(cls, ref_dict: dict) -> "MarkdownReference":
        return cls(type=ref_dict["type"], file=FileModel.from_dict(ref_dict["file"]))


@dataclass
class JsonReference(BaseReference):
    file: FileModel

    @classmethod
    def from_openapi(cls, ref_dict: dict) -> "JsonReference":
        return cls(type=ref_dict["type"], file=FileModel.from_dict(ref_dict["file"]))


class TypedReference:
    @classmethod
    def from_openapi(cls, d: dict) -> RefType:
        type_ = d["type"]
        ref_map = {
            "text": TextReference,
            "doc_x": DocxReference,
            "pdf": PdfReference,
            "markdown": MarkdownReference,
            "json": JsonReference,
        }
        return ref_map[type_].from_openapi(d)


@dataclass
class Snippet(BaseDataclass):
    type: str
    content: str
    score: float
    reference: RefType

    @classmethod
    def from_openapi(cls, snippet: OpenAPISnippetModel):
        return cls(
            type=snippet.type,
            content=snippet.content,
            score=snippet.score,
            reference=TypedReference.from_openapi(snippet.reference),
        )


@dataclass
class ContextResponse(BaseDataclass):
    id: str
    snippets: list[Snippet]
    usage: TokenCounts

    @classmethod
    def from_openapi(cls, ctx: OpenAPIContextModel):
        return cls(
            id=ctx.id,
            snippets=[Snippet.from_openapi(snippet) for snippet in ctx.snippets],
            usage=TokenCounts.from_openapi(ctx.usage),
        )
