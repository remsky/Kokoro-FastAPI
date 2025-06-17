from enum import Enum


class CloudProvider(Enum):
    """Cloud providers available for use with Pinecone serverless indexes

    This list could expand or change over time as more cloud providers are supported.
    Check the Pinecone documentation for the most up-to-date list of supported cloud
    providers. If you want to use a cloud provider that is not listed here, you can
    pass a string value directly without using this enum.
    """

    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class AwsRegion(Enum):
    """AWS (Amazon Web Services) regions available for use with Pinecone serverless indexes

    This list could expand or change over time as more regions are supported.
    Check the Pinecone documentation for the most up-to-date list of supported
    regions. If you want to use a region that is not listed here, you can
    pass a string value directly without using this enum.
    """

    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"


class GcpRegion(Enum):
    """GCP (Google Cloud Platform) regions available for use with Pinecone serverless indexes

    This list could expand or change over time as more regions are supported.
    Check the Pinecone documentation for the most up-to-date list of supported
    regions. If you want to use a region that is not listed here, you can
    pass a string value directly without using this enum.
    """

    US_CENTRAL1 = "us-central1"
    EUROPE_WEST4 = "europe-west4"


class AzureRegion(Enum):
    """Azure regions available for use with Pinecone serverless indexes

    This list could expand or change over time as more regions are supported.
    Check the Pinecone documentation for the most up-to-date list of supported
    regions. If you want to use a region that is not listed here, you can
    pass a string value directly without using this enum.
    """

    EASTUS2 = "eastus2"
