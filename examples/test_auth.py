#!/usr/bin/env python
"""
Test script for Kokoro TTS API authentication
"""

import argparse
import json
import os
import sys
from typing import Optional

import requests


def test_auth(base_url: str, api_key: Optional[str] = None) -> None:
    """Test authentication with the API"""
    # Test the test endpoint
    test_url = f"{base_url}/v1/test"
    test_response = requests.get(test_url)
    test_data = test_response.json()
    
    print(f"Test endpoint response: {json.dumps(test_data, indent=2)}")
    print(f"Authentication enabled: {test_data.get('auth_enabled', False)}")
    print(f"API keys configured: {test_data.get('api_keys_configured', False)}")
    
    # Test the models endpoint
    models_url = f"{base_url}/v1/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    print("\nTesting models endpoint...")
    models_response = requests.get(models_url, headers=headers)
    
    if models_response.status_code == 200:
        print("✅ Authentication successful!")
        models_data = models_response.json()
        print(f"Available models: {', '.join([model['id'] for model in models_data.get('data', [])])}")
    elif models_response.status_code == 401:
        print("❌ Authentication failed: Unauthorized")
        print(f"Error details: {models_response.json()}")
    else:
        print(f"❌ Unexpected response: {models_response.status_code}")
        print(f"Response: {models_response.text}")


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser(description="Test Kokoro TTS API authentication")
    parser.add_argument("--url", default="http://localhost:8880", help="Base URL of the API")
    parser.add_argument("--key", help="API key to use for authentication")
    
    args = parser.parse_args()
    
    # Use environment variable if key not provided
    api_key = args.key or os.environ.get("KOKORO_API_KEY")
    
    test_auth(args.url, api_key)


if __name__ == "__main__":
    main() 