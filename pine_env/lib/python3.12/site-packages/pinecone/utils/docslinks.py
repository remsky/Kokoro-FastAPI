def versioned_url(template: str):
    return lambda version: template.format(version)


docslinks = {
    "README": "https://github.com/pinecone-io/pinecone-python-client/blob/main/README.md",
    "GITHUB_REPO": "https://github.com/pinecone-io/pinecone-python-client",
    "LANGCHAIN_IMPORT_KB_ARTICLE": "https://docs.pinecone.io/troubleshooting/pinecone-attribute-errors-with-langchain",
    "API_DESCRIBE_INDEX": versioned_url(
        "https://docs.pinecone.io/reference/api/{}/control-plane/describe_index"
    ),
}
