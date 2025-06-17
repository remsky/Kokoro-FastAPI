from pinecone.utils import docslinks

KB_ARTICLE = docslinks["LANGCHAIN_IMPORT_KB_ARTICLE"]
GITHUB_REPO = docslinks["GITHUB_REPO"]


def _build_langchain_attribute_error_message(method_name: str):
    return f"""{method_name} is not a top-level attribute of the Pinecone class provided by pinecone's official python package developed at {GITHUB_REPO}. You may have a name collision with an export from another dependency in your project that wraps Pinecone functionality and exports a similarly named class. Please refer to the following knowledge base article for more information: {KB_ARTICLE}
"""
