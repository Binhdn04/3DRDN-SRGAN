
IMAGE_NAME = dl-app
TAG = latest
CONTAINER_NAME = dl-container

build:
	docker build -t $(IMAGE_NAME):$(TAG) .

run:
	docker run -it --gpus all --rm \
		-v $(PWD):/app \
		-p 8888:8888 \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME):$(TAG)

run-bg:
	docker run -d --gpus all \
		-v $(PWD):/app \
		-p 8888:8888 \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME):$(TAG) \
		sleep infinity

exec:
	docker exec -it $(CONTAINER_NAME) bash

stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

clean: stop
	docker rmi $(IMAGE_NAME):$(TAG) || true

.PHONY: build run run-bg exec stop clean
