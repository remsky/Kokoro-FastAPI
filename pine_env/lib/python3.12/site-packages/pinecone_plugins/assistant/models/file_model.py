from dataclasses import dataclass
from typing import Dict, Any, Optional
from pinecone_plugins.assistant.data.core.client.model.assistant_file_model import AssistantFileModel as OpenAIFileModel
from pinecone_plugins.assistant.models.core.dataclass import BaseDataclass


@dataclass
class FileModel(BaseDataclass):
    name: str
    id: str
    metadata: Dict[str, Any]
    created_on: str
    updated_on: str
    status: str
    percent_done: float
    signed_url: str
    error_message: Optional[str]
    size: int

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            name=d.get("name"),
            id=d.get("id"),
            metadata=d.get("metadata"),
            created_on=d.get("created_on"),
            updated_on=d.get("updated_on"),
            status=d.get("status"),
            percent_done=d.get("percent_done"),
            signed_url=d.get("signed_url"),
            error_message=d.get("error_message"),
            size=d.get("size")
        )

    @classmethod
    def from_openapi(cls, file_model: OpenAIFileModel):
        return cls(
            name=file_model.name,
            id=file_model.id,
            metadata=file_model.metadata,
            created_on=file_model.created_on,
            updated_on=file_model.updated_on,
            status=file_model.status,
            percent_done=file_model.percent_done,
            signed_url=file_model.signed_url,
            error_message=file_model.error_message,
            size=file_model.size
        )
