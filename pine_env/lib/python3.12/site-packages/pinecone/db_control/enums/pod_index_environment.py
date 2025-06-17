from enum import Enum


class PodIndexEnvironment(Enum):
    """
    These environment strings are used to specify where a pod index should be deployed.
    """

    US_WEST1_GCP = "us-west1-gcp"
    US_CENTRAL1_GCP = "us-central1-gcp"
    US_WEST4_GCP = "us-west4-gcp"
    US_EAST4_GCP = "us-east4-gcp"
    NORTHAMERICA_NORTHEAST1_GCP = "northamerica-northeast1-gcp"
    ASIA_NORTHEAST1_GCP = "asia-northeast1-gcp"
    ASIA_SOUTHEAST1_GCP = "asia-southeast1-gcp"
    US_EAST1_GCP = "us-east1-gcp"
    EU_WEST1_GCP = "eu-west1-gcp"
    EU_WEST4_GCP = "eu-west4-gcp"
    US_EAST1_AWS = "us-east-1-aws"
    EASTUS_AZURE = "eastus-azure"
