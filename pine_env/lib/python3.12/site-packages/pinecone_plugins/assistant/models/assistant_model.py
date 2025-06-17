import time
from io import BytesIO
from urllib.parse import urljoin

import requests
import json
from typing import Iterable, List, Optional, Union, Dict, TypeVar, BinaryIO

from pinecone_plugins.assistant.data.core.client.api.manage_assistants_api import (
    ManageAssistantsApi as DataApiClient,
)
from pinecone_plugins.assistant.data.core.client.model.search_completions import (
    SearchCompletions as ChatCompletionsRequest,
)
from pinecone_plugins.assistant.data.core.client.model.chat import Chat as ChatRequest
from pinecone_plugins.assistant.data.core.client.model.message_model import MessageModel
from pinecone_plugins.assistant.data.core.client.model.context_options_model import ContextOptionsModel
from pinecone_plugins.assistant.control.core.client.models import (
    Assistant as OpenAIAssistantModel,
)
from pinecone_plugins.assistant.data.core.client import ApiClient
from pinecone_plugins.assistant.models.file_model import FileModel

from .chat import (
    ContextOptions,
    Message,
    StreamChatResponseCitation,
    StreamChatResponseContentDelta,
    StreamChatResponseMessageEnd,
    StreamChatResponseMessageStart,
    ChatResponse, BaseStreamChatResponseChunk
)
from .chat_completion import StreamingChatCompletionChunk, ChatCompletionResponse
from .context_responses import ContextResponse
from ..data.core.client.model.context_request import ContextRequest

RawMessage = Dict
RawMessages = Union[List[Message], List[RawMessage]]
S = TypeVar("S", bound=BaseStreamChatResponseChunk)
HOST_SUFFIX = "assistant"
MODELS = ["gpt-4o", "gpt-4.1", "o4-mini", "claude-3-5-sonnet", "claude-3-7-sonnet", "gemini-2.5-pro"]


class AssistantModel:
    def __init__(self, assistant: OpenAIAssistantModel, client_builder, config):
        self.assistant = assistant
        self.host = assistant.host
        self.host = urljoin(self.host, HOST_SUFFIX)
        self.config = config if config else {}

        self._assistant_data_api = client_builder(
            ApiClient, DataApiClient, "2025-01", host=self.host
        )
        # initialize types so they can be accessed
        self.name = self.assistant.name
        self.created_at = self.assistant.created_at
        self.updated_at = self.assistant.updated_at
        self.metadata = self.assistant.metadata
        self.status = self.assistant.status
        self.ctxs = []

    def __str__(self):
        return str(self.assistant)

    def __repr__(self):
        return repr(self.assistant)

    def __getattr__(self, attr):
        return getattr(self.assistant, attr)

    def upload_file(
        self,
        file_path: str,
        metadata: Optional[dict[str, any]] = None,
        timeout: Optional[int] = None,
    ) -> FileModel:
        """
        Uploads a file from the specified path to this assistant for internal processing.

        :param file_path: The path to the file that needs to be uploaded.
        :type file_path: str, required

        :param metadata: Optional metadata dictionary to be attached to the file.
        :type metadata: Optional[dict[str, any]], optional

        :param timeout: Specify the number of seconds to wait until file processing is done. If None, wait indefinitely; if >=0, time out after this many seconds;
            if -1, return immediately and do not wait. Default: None
        :type timeout: int, optional


        :return: FileModel object with the following properties:
            - id: The UUID of the uploaded file.
            - name: The name of the uploaded file.
            - created_on: The timestamp of when the file was created.
            - updated_on: The timestamp of the last update to the file.
            - metadata: Metadata associated with the file.
            - status: The status of the file.

        Example:
        >>> assistant = (...).assistant.Assistant("assistant_name")
        >>> file_model = assistant.upload_file(file_path="/path/to/file.txt") # use the default timeout
        >>> print(file_model)
          {'created_on': '2024-06-02T19:48:00Z',
          'id': '070513b3-022f-4966-b583-a9b12e0920ff',
          'metadata': None,
          'name': 'tiny_file.txt',
          'status': 'Available',
          'updated_on': '2024-06-02T19:48:00Z'}
        """
        try:
            with open(file_path, "rb") as file:
                return self._upload_file_stream(file, metadata, timeout)
        except FileNotFoundError:
            raise Exception(f"Error: The file at {file_path} was not found.")
        except IOError:
            raise Exception(f"Error: Could not read the file at {file_path}.")

    def upload_bytes_stream(
            self,
            stream: BytesIO,
            file_name: str,
            metadata: Optional[dict[str, any]] = None,
            timeout: Optional[int] = None,
    ) -> FileModel:
        """
        Uploads a file-like stream to the assistant for internal processing.

        Note: for text files, the stream must be encoded in UTF-8.

        :param stream: BytesIO stream containing the file bytes to be uploaded.
        :type stream: BytesIO, required

        :param file_name: The file name to associate with the stream.
        :type file_name: str, required

        :param metadata: Optional metadata dictionary to be attached to the file.
        :type metadata: Optional[dict[str, any]], optional

        :param timeout: Specify the number of seconds to wait until file processing is done. If None, wait indefinitely; if >=0, time out after this many seconds;
            if -1, return immediately and do not wait. Default: None
        :type timeout: int, optional

        :return: FileModel object representing the uploaded file.

        Example:
        >>> assistant = (...).assistant.Assistant("assistant_name")
        Example:
        >>> from io import BytesIO
        >>> # Create a BytesIO stream with some binary data.
        >>> text = "Hello, world!"
        >>> stream = BytesIO(text.encode("utf-8"))
        >>> # Instantiate your assistant object (assuming proper initialization).
        >>> assistant = Assistant("assistant_name")
        >>> # Upload the stream with a specified file name.
        >>> file_model = assistant.upload_bytes_stream(stream, "myfile.txt")
        >>> print(file_model)
        {'created_on': '2024-06-02T19:48:00Z',
         'id': '070513b3-022f-4966-b583-a9b12e0920ff',
         'metadata': None,
         'name': 'myfile.txt',
         'status': 'Available',
         'updated_on': '2024-06-02T19:48:00Z'}
        """
        stream.name = file_name
        return self._upload_file_stream(stream, metadata, timeout)


    def _upload_file_stream(
        self,
        file_stream: BinaryIO,
        metadata: Optional[dict[str, any]] = None,
        timeout: Optional[int] = None,
    ) -> FileModel:
            upload_resp = (
                self._assistant_data_api.upload_file(
                    assistant_name=self.assistant.name,
                    file=file_stream,
                    metadata=json.dumps(metadata),
                )
                if metadata
                else self._assistant_data_api.upload_file(
                    assistant_name=self.assistant.name, file=file_stream
                )
            )

            # wait for status
            if timeout == -1:
                # still in processing state
                return FileModel.from_openapi(upload_resp)
            if timeout is None:
                while upload_resp.status == "Processing":
                    time.sleep(5)
                    upload_resp = self.describe_file(upload_resp.id)
                    if upload_resp.status == "ProcessingFailed":
                        raise Exception(f"File processing failed. Error: {upload_resp.error_message}")
            else:
                while upload_resp.status == "Processing" and timeout >= 0:
                    time.sleep(5)
                    timeout -= 5
                    upload_resp = self.describe_file(upload_resp.id)
                    if upload_resp.status == "ProcessingFailed":
                        raise Exception(f"File processing failed. Error: {upload_resp.error_message}")

            if timeout and timeout < 0:
                raise (
                    TimeoutError(
                        f"Please call `pc.assistant.Assistant({self.name})` to confirm assistant status."
                    )
                )
            return FileModel.from_openapi(upload_resp)

    def describe_file(self, file_id: str, include_url: Optional[bool] = False) -> FileModel:
        """
        Describes a file with the specified file_id from this assistant. Includes information on its status and metadata.

        :param : The file id of the file to be described
        :type file_id: str, required

        :param include_url: If True, the signed URL of the file is included in the response.
        :type include_url: bool, optional

        :return: FileModel object with the following properties:
            - id: The UUID of the requested file.
            - name: The name of the requested file.
            - created_on: The timestamp of when the file was created.
            - updated_on: The timestamp of the last update to the file.
            - metadata: Metadata associated with the file.
            - status: The status of the file.

        Example:
        >>> assistant = (...).assistant.Assistant("model_name")
        >>> file_model = assistant.upload_file(file_path="/path/to/file.txt") # use the default timeout
        >>> print(file_model)
          {'created_on': '2024-06-02T19:48:00Z',
          'id': '070513b3-022f-4966-b583-a9b12e0290ff',
          'metadata': None,
          'name': 'tiny_file.txt',
          'status': 'Available',
          'updated_on': '2024-06-02T19:48:00Z'}
        >>> assistant.describe_file(file_id='070513b3-022f-4966-b583-a9b12e0290ff')
          {'created_on': '2024-06-02T19:48:00Z',
          'id': '070513b3-022f-4966-b583-a9b12e0290ff',
          'metadata': None,
          'name': 'tiny_file.txt',
          'status': 'Available',
          'updated_on': '2024-06-02T19:48:00Z'}
        """

        if include_url:
            file = self._assistant_data_api.describe_file(
                assistant_name=self.name,
                assistant_file_id=file_id,
                include_url=str(include_url).lower()
            )
        else:
            file = self._assistant_data_api.describe_file(
                assistant_name=self.name,
                assistant_file_id=file_id
            )
        return FileModel.from_openapi(file)

    def list_files(self, filter: Optional[dict[str, any]] = None) -> List[FileModel]:
        """
        Lists all uploaded files in this assistant.

        :return: List of FileModel objects with the following properties:
            - id: The UUID of the requested file.
            - name: The name of the requested file.
            - created_on: The timestamp of when the file was created.
            - updated_on: The timestamp of the last update to the file.
            - metadata: Metadata associated with the file.
            - status: The status of the file.

        Example:
        >>> assistant = (...).assistant.Assistant("model_name")
        >>> assistant.list_files()
          [{'created_on': '2024-06-02T19:48:00Z',
          'id': '070513b3-022f-4966-b583-a9b12e0290ff',
          'metadata': None,
          'name': 'tiny_file.txt',
          'status': 'Available',
          'updated_on': '2024-06-02T19:48:00Z'}, ...]
        """
        files_resp = (
            self._assistant_data_api.list_files(
                self.name, filter=json.dumps(filter))
            if filter
            else self._assistant_data_api.list_files(self.name)
        )
        return [FileModel.from_openapi(file) for file in files_resp.files]

    def delete_file(self, file_id: str, timeout: Optional[int] = None):
        """
        Deletes a file with the specified file_id from this assistant.

        :param file_path: The path to the file that needs to be uploaded.
        :type file_path: str, required

        :type timeout: int, optional
        :param timeout: Specify the number of seconds to wait until file processing is done. If None, wait indefinitely; if >=0, time out after this many seconds;
            if -1, return immediately and do not wait. Default: None

        Example:
        >>> assistant = (...).assistant.Assistant("model_name")
        >>> assistant.delete_file(file_id='070513b3-022f-4966-b583-a9b12e0290ff') # use the default timeout
        >>> assistant.list_files()
          []
        """
        self._assistant_data_api.delete_file(
            assistant_name=self.name, assistant_file_id=file_id
        )

        if timeout == -1:
            # still in processing state
            return
        if timeout is None:
            file = self.describe_file(file_id=file_id)
            while file:
                time.sleep(5)
                try:
                    file = self.describe_file(file_id=file_id)
                except Exception:
                    file = None
        else:
            file = self.describe_file(file_id=file_id)
            while file and timeout >= 0:
                time.sleep(5)
                timeout -= 5
                try:
                    file = self.describe_file(file_id=file_id)
                except Exception:
                    file = None

        if timeout and timeout < 0:
            raise (
                TimeoutError(
                    "Please call the describe_model API ({}) to confirm model status.".format(
                        "https://www.pinecone.io/docs/api/operation/assistant/describe_model/"
                    )
                )
            )

    @classmethod
    def _parse_messages(cls, messages: Union[List[Message], List[RawMessage]]) -> List[Message]:
        return [Message.from_dict(message) if isinstance(message, dict) else message for message in messages]

    def chat_completions(
        self,
        messages: Union[List[Message], List[RawMessage]],
        filter: Optional[dict[str, any]] = None,
        stream: bool = False,
        model: Union[str, None] = None,
        temperature: Optional[float] = None,
    ) -> Union[ChatCompletionResponse, Iterable[StreamingChatCompletionChunk]]:
        """
        Performs a chat completion request to the following assistant. Use this method if you want the response output to be OpenAI's chat completion format.

        :param messages: The current context for the chat request. The final element in the list represents the user query to be made from this context.
        :type messages: List[Message] where Message requires the following:
            Message:
                - role: str, the role of the context ('user' or 'agent')
                - content: str, the content of the context

            Alternatively, you can pass a list of dictionaries with the following keys:
                - role: str, the role of the context ('user' or 'agent')
                - content: str, the content of the context

        :param model: The large language model to use for answer generation. If this flag is set to 'None', then the model used is OpenAI's GPT-4o.
        :type model: str | None (default 'gpt-4o' in case `None` is passed)

        :param temperature: Controls the randomness of the model's output: lower values make responses more deterministic, while higher values increase creativity and variability. If the model does not support a temperature parameter, the parameter will be ignored.
        :type temperature: float | None (default 0.0 in case `None` is passed)
        
        :param filter: Optional dictionary to filter which documents can be used in this query.
                       Use this to narrow down the context for the assistant's response.
        :type filter: Optional[dict[str, any]] (default None)

        Example filter:
            {
                "genre": {"$ne": "documentary"}
            }
        This filter would exclude documents with the genre "documentary" from being used in the query.

        :param stream: If this flag is turned on, then the return type is an Iterable[StreamingChatCompletionChunk] whether data is returned as a generator/stream
        :type stream: bool (default false)

        :return:
        The default result is a ChatCompletionResponse with the following format:
            {
                "choices": [
                    {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "The 2020 World Series was played in Texas at Globe Life Field in Arlington.",
                        "role": "assistant"
                    },
                    "logprobs": null
                    }
                ],
                "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
                "model": "gpt-3.5-turbo-0613",
            }

        However, when stream is set to true, the response is an iterable of StreamingChatCompletionChunks. See examples below:
            {
                "choices": [
                    {
                    "finish_reason": null,
                    "index": 0,
                    "delta": {
                        "content": "The",
                        "role": ""
                    },
                    "logprobs": null
                    }
                ],
                "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
                "model": "gpt-3.5-turbo-0613",
            }

        Example:
        >>> from pinecone_plugins.assistant.models import Message
        >>> assistant = (...).assistant.Assistant("planets-km")
        >>> messages = [Message(role='user', content='How old is the earth')]
        >>> resp = assistant.chat_completions(messages=messages)
        >>> print(resp)
        {'choices': [{'finish_reason': 'stop',
              'index': 0,
              'message': {'content': 'The age of the Earth is estimated to be '
                                     'about 4.54 billion years, based on '
                                     'evidence from radiometric age dating of '
                                     'meteorite material and Earth rocks, as '
                                     'well as lunar samples. This estimate has '
                                     'a margin of error of about 1%.',
                          'role': 'assistant'}}],
        'id': 'chatcmpl-9VmkSD9s7rfP28uScLlheookaSwcB',
        'model': 'planets-km'}

        Streaming example:
        >>> resp = assistant.chat_completions(messages=messages, stream=True)
        >>> for chunk in resp:
                if chunk:
                    print(chunk)

        [{'choices': [{'finish_reason': 'stop',
              'index': 0,
              'delta': {'content': 'The age of the Earth is estimated to be '
                                     'about 4.54 billion years, based on '
                                     'evidence from radiometric age dating of '
                                     'meteorite material and Earth rocks, as '
                                     'well as lunar samples. This estimate has '
                                     'a margin of error of about 1%.',
                          'role': 'assistant'}}],
        'id': 'chatcmpl-9VmkSD9s7rfP28uScLlheookaSwcB',
        'model': 'gpt-4o'}, ... ]

        """
        if model is None:
            model = "gpt-4o"
        elif model not in MODELS:
            raise ValueError(
                f"Invalid model. Valid options are {', '.join(f'`{m}`' for m in MODELS)}."
            )
        messages = self._parse_messages(messages)

        if stream:
            return self._chat_completions_streaming(
                messages=messages, model=model, filter=filter, temperature=temperature
            )
        else:
            return self._chat_completions_single(
                messages=messages, model=model, filter=filter, temperature=temperature
            )

    def _chat_completions_single(
        self,
        messages: List[Message],
        model: str = "gpt-4o",
        filter: dict[str, any] = None,
        temperature: Optional[float] = None,
    ) -> ChatCompletionResponse:
        messages = [
            MessageModel(role=ctx.role, content=ctx.content) for ctx in messages
        ]

        kwargs = {"messages": messages, "model": model}
        if filter:
            kwargs["filter"] = filter
        if temperature is not None:
            kwargs["temperature"] = temperature

        chat_request = ChatCompletionsRequest(**kwargs)
        
        result = self._assistant_data_api.chat_completion_assistant(
            assistant_name=self.name, search_completions=chat_request
        )
        return ChatCompletionResponse.from_openapi(result)

    def _chat_completions_streaming(
        self,
        messages: List[Message],
        model: str = "gpt-4o",
        filter: Optional[dict[str, any]] = None,
        temperature: Optional[float] = None,
    ) -> Iterable[StreamingChatCompletionChunk]:
        api_key = self.config.api_key
        base_url = f"{self.host}/chat/{self.name}/chat/completions"
        headers = {"api-key": api_key, "Content-Type": "application/json"}
        messages = [vars(message) for message in messages]
        content = {"messages": messages, "stream": True, "model": model}
        if filter:
            content["filter"] = filter
        if temperature is not None:
            content["temperature"] = temperature

        try:
            response = requests.post(
                base_url, headers=headers, json=content, timeout=60, stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = line.decode("utf-8")
                    if data.startswith("data:"):
                        data = data[5:]

                    json_data = json.loads(data)
                    res = StreamingChatCompletionChunk.from_dict(json_data)

                    yield res
        except Exception as e:
            raise ValueError(f"Error in chat completions streaming: {e}")

    def chat(
        self,
        messages: Union[List[Message], List[RawMessage]],
        filter: Optional[dict[str, any]] = None,
        stream: bool = False,
        model: Union[str, None] = None,
        temperature: Optional[float] = None,
        json_response: bool = False,
        include_highlights: bool = False,
        context_options: Optional[Union[ContextOptions, dict[str, int]]] = None,
    ) -> Union[ChatResponse, Iterable[S]]:
        """
        Performs a chat request to the following assistant.

        :param messages: The current context for the chat request. The final element in the list represents the user query to be made from this context.
        :type messages: List[Message] where Message requires the following:
            Message:
                - role: str, the role of the context ('user' or 'agent')
                - content: str, the content of the context

            Alternatively, you can pass a list of dictionaries with the following keys:
                - role: str, the role of the context ('user' or 'agent')
                - content: str, the content of the context

        :param model: The large language model to use for answer generation. If this flag is set to 'None', then the model used is OpenAI's GPT-4o.
        :type model: str | None (default 'gpt-4o' in case `None` is passed)

        :param temperature: Controls the randomness of the model's output: lower values make responses more deterministic, while higher values increase creativity and variability. If the model does not support a temperature parameter, the parameter will be ignored.
        :type temperature: float | None (default 0.0 in case `None` is passed)
        
        :param json_response: If true, the assistant will be instructed to return a JSON response. Cannot be used with streaming.
        :type json_response: bool (default False)

        :param include_highlights: If true, the assistant will be instructed to mark highlights for citations. Highlights are small snippets of text that are relevant to the citation.
        :type include_highlights: bool (default False)

        :param filter: Optional dictionary to filter which documents can be used in this query.
                       Use this to narrow down the context for the assistant's response.
        :type filter: Optional[dict[str, any]] (default None)

        Example filter:
            {
                "genre": {"$ne": "documentary"}
            }
        This filter would exclude documents with the genre "documentary" from being used in the query.

        :param stream: If this flag is turned on, then the return type is an Iterable[StreamChatResponseChunk] whether data is returned as a generator/stream
        :type stream: bool (default false)

        :param context_options: Option to control the context snippets sent to the LLM.
        :type context_options: Optional[ContextOptions] (default None)
            ContextOptions:
                - top_k: int, the number of context snippets to use. Default is 16. Maximum is 64.
                - snippet_size: int, the maximum context snippet size. Default is 2048 tokens. Minimum is 512 tokens. Maximum is 8192 tokens.
            
            Alternatively, you can pass a dictionary with the following keys:
                - top_k: int, the number of context snippets to use.
                - snippet_size: int, the maximum context snippet size.

        :return:
        The default result is a ChatModel with the following format:
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The 2020 World Series was played in Texas at Globe Life Field in Arlington.",
                    "role": "assistant"
                },
                "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
                "model": "gpt-3.5-turbo-0613",
                "citations": [
                    {
                        "position": 3,
                        "references": [
                            {
                                "file": {
                                    'created_on': '2024-06-02T19:48:00Z',
                                    'id': '070513b3-022f-4966-b583-a9b12e0290ff',
                                    'metadata': None,
                                    'name': 'tiny_file.txt',
                                    'status': 'Available',
                                    'updated_on': '2024-06-02T19:48:00Z'
                                },
                                "pages": [1, 2, 3],
                                "highlight": Null,
                            }
                        ],
                    }
                ]
            }

        However, when stream is set to true, the response is an stream of StreamChatResponseChunks. This can be one of the following types:
        - StreamChatResponseMessageStart:
            {'type': 'message_start', 'id': '0000000000000000468323be9d266e55', 'model': 'gpt-4o-2024-11-20', 'role': 'assistant'}
        - StreamChatResponseContentDelta
            {'type': 'content_chunk', 'id': '0000000000000000468323be9d266e55', 'model': 'gpt-4o-2024-11-20', 'delta': {'content': 'The'}}
        - StreamChatResponseCitation
            {'type': 'citation', 'id': '0000000000000000116990b44044d21e', 'model': 'gpt-4o-2024-11-20', 'citation': {'position': 247, 'references': [{'id': 's0', 'file': {'status': 'Available', 'id': '985edb6c-f649-4334-8f14-9a16b7039ab6',
            'name': 'PEPSICO_2022_10K.pdf', 'size': 2993516, 'metadata': None, 'updated_on': '2024-08-08T15:41:58.839846634Z', 'created_on': '2024-08-08T15:41:07.427879083Z', 'percent_done': 0.0,
            'signed_url': 'example.com'}, 'pages': [32]}]}}
        - StreamChatResponseMessageEnd
            {'type': 'message_end', 'id': '0000000000000000116990b44044d21e', 'model': 'gpt-4o-2024-11-20', 'finish_reason': 'stop', 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}


        Example:
        >>> from pinecone_plugins.assistant.models import Message
        >>> assistant = (...).assistant.Assistant("planets-km")
        >>> messages = [Message(role='user', content='How old is the earth')]
        >>> resp = assistant.chat(messages=messages, context_options=ContextOptions(top_k=10, snippet_size=4096))
        >>> print(resp)
        {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                   'content': 'The age of the Earth is estimated to be '
                                     'about 4.54 billion years, based on '
                                     'evidence from radiometric age dating of '
                                     'meteorite material and Earth rocks, as '
                                     'well as lunar samples. This estimate has '
                                     'a margin of error of about 1%.',
                    'role': 'assistant'
                },
                'id': 'chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW',
                "model": "gpt-3.5-turbo-0613",
                "citations": [
                    {
                        "position": 3,
                        "references": [
                            {
                                "file": {
                                    'created_on': '2024-06-02T19:48:00Z',
                                    'id': '070513b3-022f-4966-b583-a9b12e0290ff',
                                    'metadata': None,
                                    'name': 'tiny_file.txt',
                                    'status': 'Available',
                                    'updated_on': '2024-06-02T19:48:00Z'
                                },
                                "pages": [1, 2, 3],
                                "highlight": Null,
                            }
                        ],
                    }
                ]
            }

        Streaming example:
        >>> resp = assistant.chat(messages=messages, stream=True, context_options={"top_k":10, "snippet_size":4096})
        >>> for chunk in resp:
                if chunk:
                    print(chunk)

        [{'type': 'message_start', 'id': '0000000000000000468323be9d266e55', 'model': 'gpt-4o-2024-11-20', 'role': 'assistant'},
         {'type': 'content_chunk', 'id': '0000000000000000468323be9d266e55', 'model': 'gpt-4o-2024-11-20', 'delta': {'content': 'The'}},
          ...
         {'type': 'message_end', 'id': '0000000000000000116990b44044d21e', 'model': 'gpt-4o-2024-11-20', 'finish_reason': 'stop', 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}]

        """
        if model is None:
            model = "gpt-4o"
        elif model not in MODELS:
            raise ValueError(
                f"Invalid model. Valid options are {', '.join(f'`{m}`' for m in MODELS)}."
            )
        if json_response and stream:
            raise ValueError("Cannot use json_response with streaming")

        messages = self._parse_messages(messages)
        context_options = ContextOptions.from_dict(context_options) if isinstance(context_options, dict) else context_options

        if stream:
            return self._chat_streaming(messages=messages, model=model, filter=filter, include_highlights=include_highlights, context_options=context_options, temperature=temperature)
        else:
            return self._chat_single(messages=messages, model=model, filter=filter, json_response=json_response, include_highlights=include_highlights, context_options=context_options, temperature=temperature)

    def _chat_single(
        self,
        messages: List[Message],
        model: str = "gpt-4o",
        temperature: Optional[float] = None,
        filter: dict[str, any] = None,
        json_response: bool = False,
        include_highlights: bool = False,
        context_options: Optional[ContextOptions] = None,
    ) -> ChatResponse:
        messages = [
            MessageModel(role=ctx.role, content=ctx.content) for ctx in messages
        ]

        kwargs = {
            "messages": messages,
            "model": model,
            "json_response": json_response,
            "include_highlights": include_highlights,
        }
        if filter:
            kwargs["filter"] = filter
        if temperature is not None:
            kwargs["temperature"] = temperature
        if context_options is not None:
            options = {}
            if context_options.top_k is not None:
                options["top_k"] = context_options.top_k
            if context_options.snippet_size is not None:
                options["snippet_size"] = context_options.snippet_size
            if options:
                kwargs["context_options"] = ContextOptionsModel(**options)

        chat_request = ChatRequest(**kwargs)
        chat_result = self._assistant_data_api.chat_assistant(
            assistant_name=self.name, chat=chat_request
        )
        return ChatResponse.from_openapi(chat_result)

    def _chat_streaming(
        self,
        messages: List[Message],
        model: str = "gpt-4o",
        temperature: Optional[float] = None,
        filter: Optional[dict[str, any]] = None,
        include_highlights: bool = False,
        context_options: Optional[ContextOptions] = None,
    ) -> Iterable[S]:
        api_key = self.config.api_key
        base_url = f"{self.host}/chat/{self.name}"
        headers = {"api-key": api_key, "Content-Type": "application/json"}
        messages = [vars(message) for message in messages]
        content = {"messages": messages, "stream": True, "model": model, "include_highlights": include_highlights}

        if filter:
            content["filter"] = filter
        if temperature is not None:
            content["temperature"] = temperature
        if context_options is not None:
            options = {}
            if context_options.top_k is not None:
                options["top_k"] = context_options.top_k
            if context_options.snippet_size is not None:
                options["snippet_size"] = context_options.snippet_size
            if options:
                content["context_options"] = options

        try:
            response = requests.post(
                base_url, headers=headers, json=content, timeout=60, stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = line.decode("utf-8")
                    if data.startswith("data:"):
                        data = data[5:]

                    json_data = json.loads(data)

                    res = None
                    if json_data.get("type") == "message_start":
                        res = StreamChatResponseMessageStart.from_dict(
                            json_data)
                    elif json_data.get("type") == "content_chunk":
                        res = StreamChatResponseContentDelta.from_dict(
                            json_data)
                    elif json_data.get("type") == "citation":
                        res = StreamChatResponseCitation.from_dict(json_data)
                    elif json_data.get("type") == "message_end":
                        res = StreamChatResponseMessageEnd.from_dict(json_data)

                    yield res
        except Exception as e:
            raise ValueError(f"Error in chat completions streaming: {e}")

    def context(
        self,
        query: Optional[str] = None,
        messages: Union[List[Message], List[RawMessage]] = None,
        filter: Optional[dict[str, any]] = None,
        top_k: Optional[int] = None,
        snippet_size: Optional[int] = None,
    ):
        """
        Performs a context request to the following assistant.

        :param query: The query to be used in the context request. Either
                      one of query or messages may be used to generate
                      the context response.
        :type query: Optional[str]

        :param messages: The messages to be used in the context request. Either
                         one of query or messages may be used to generate the
                         context response.
        :type messages: Optional[List[Message]]

        :param filter: Optional dictionary to filter which documents can be used in this query.
                       Use this to narrow down the context for the assistant's response.
        :type filter: Optional[dict[str, any]] (default None)

        Example filter:
            {
                "genre": {"$ne": "documentary"}
            }
        This filter would exclude documents with the genre "documentary" from being used in the query.

        :param top_k: Optional integer to specify the maximum number of context snippets to return.
        :type top_k: Optional[int] (default None)

        :param snippet_size: Optional integer to specify the maximum context snippet size. Default is 2048 tokens. Minimum is 512 tokens. Maximum is 8192 tokens.
        :type snippet_size: Optional[int] (default None)
        
        :return:
        The default result is a ContextResponse with the following format:

        {
          "snippets": [
            {
              "type": "text",
              "content": "The quick brown fox jumps over the lazy dog.",
              "score": 0.9946,
              "reference": {
                "type": "pdf",
                "file": {
                  "id": "96e6e2de-82b2-494d-8988-7dc88ce2ac01",
                  "metadata": null,
                  "name": "sample.pdf",
                  "percent_done": 1.0,
                  "status": "Available",
                  "created_on": "2024-11-13T14:59:53.369365582Z",
                  "updated_on": "2024-11-13T14:59:55.369365582Z",
                  "signed_url": "https://storage.googleapis.com/..."
                },
                "pages": [1]
              }
            }
          ],
          "usage": {
            "completion_tokens": 0,
            "prompt_tokens": 506,
            "total_tokens": 506
          }
        }

        Example:
        >>> assistant = (...).assistant.Assistant("planets-km")
        >>> resp = assistant.context(query="What is the age of the earth?")
        >>> print(resp)
        """
        if not ((not query and messages) or (not messages and query)):
            print(query, messages)
            return ValueError("Invalid Inputs: Exactly one of query or messages must be inputted.")

        kwargs = {}
        if messages:
            messages = self._parse_messages(messages)
            messages = [MessageModel(role=ctx.role, content=ctx.content) for ctx in messages]
            kwargs["messages"] = messages
        else:
            kwargs["query"] = query

        if filter:
            kwargs["filter"] = filter
        if top_k is not None:
            kwargs["top_k"] = top_k
        if snippet_size is not None:
            kwargs["snippet_size"] = snippet_size

        context_request = ContextRequest(
            **kwargs
        )
        raw_response = self._assistant_data_api.context_assistant(
            assistant_name=self.name,
            context_request=context_request
        )
        return ContextResponse.from_openapi(raw_response)
