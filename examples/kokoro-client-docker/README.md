The example to run client in other Container.

## Build Docker Image for the Client

```bash
cd examples/kokoro-client-docker
sudo UID=${UID} GID=$(id -g) docker-compose up -d

sudo docker exec -it kokoro-client bash
```


## Run command inside Container

- Save audio to file

```bash
python save_to_file.py \
--base_url http://host.docker.internal:8880/v1 \
--text_input "Hi, how are you?"
```

- Stream to speaker

```bash
python stream_to_speaker.py \
--base_url http://host.docker.internal:8880/v1 \
--text_input "Hi, how are you?"
```