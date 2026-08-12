
## Kubernetes Helm chart installation with GPU

*Last updated: 2026-08-11*
> Thanks to [@zucher](https://github.com/zucher) for the PR and Wiki guide [Add Helm chart #157](https://github.com/remsky/Kokoro-FastAPI/pull/157)
## Installation Guide

### 1. Prerequisites
Ensure you have the following tools installed:
- **Helm**: [Install Helm](https://helm.sh/docs/intro/install/)
- **kubectl**: Configured to communicate with your Kubernetes cluster
- **Git**: Installed on your system

> Verify the Nvidia GPU operator is correctly installed on your Kubernetes cluster ([see Installing the NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/24.9.2/getting-started.html))
  

### 2. Clone the GitHub Repository
Clone the repository containing the Kokoro-FastAPI Helm chart.
```bash
git clone https://github.com/remsky/Kokoro-FastAPI.git
cd Kokoro-FastAPI/charts/kokoro-fastapi
```

### 3. Install the Chart
Navigate to the directory with the Helm chart and install it using Helm.

#### Installation in the `target-namespace` of you choice
```bash
helm install kokoro-fastapi . --namespace <target-namespace> --create-namespace
```

This command installs the chart in the `target-namespace` namespace, creating the namespace if it doesn't exist.

### 4. Verify Installation
Check that your application is running by listing the resources in the namespace.
```bash
kubectl get all -n <target-namespace>
```

## Customizing the Helm Chart

If you need to customize the installation, you can use a custom `values.yaml` file or override specific values directly via command-line arguments.

### Using a Custom `values.yaml` File
Create your own `values.yaml` file with custom configurations and install the chart using it:
```bash
helm install kokoro-fastapi . --namespace <target-namespace> --create-namespace -f my-custom-values.yaml
```

#### Example `my-custom-values.yaml`
Here is an example of a simple `values.yaml` file:
```yaml
kokoroTTS:
  replicaCount: 2
  repository: ghcr.io/remsky/kokoro-fastapi-gpu
  tag: latest
  pullPolicy: IfNotPresent

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: my-kokoro-endpoint.dev
      paths:
        - path: /
          pathType: Prefix

...
```

See `values.yaml` for the full set, and `examples/` for AKS and GPU operator configs.

### Overriding Values Directly
You can override specific values directly in the Helm install command:
```bash
helm install kokoro-fastapi . \
  --namespace <target-namespace> \
  --create-namespace \
  --set kokoroTTS.replicaCount=2 \
  --set kokoroTTS.tag=latest
```

## Updating the Deployment

To update your existing deployment with new configurations, use the Helm upgrade command:
```bash
helm upgrade kokoro-fastapi . -n <target-namespace> -f my-custom-values.yaml
```

Or using directly overridden values:
```bash
helm upgrade kokoro-fastapi . \
  -n <target-namespace> \
  --set kokoroTTS.replicaCount=1
```

### Important Notes

- **Rollback**: If something goes wrong, you can rollback to the previous version:
  ```bash
  helm rollback kokoro-fastapi -n <target-namespace>
  ```

