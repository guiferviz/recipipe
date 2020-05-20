
IMAGE_NAMESPACE = guiferviz
IMAGE_NAME = recipipe
IMAGE = $(IMAGE_NAMESPACE)/$(IMAGE_NAME)


all:
	cd .. && docker build -t $(IMAGE) -f docker/Dockerfile .

nocache:
	cd .. && docker build --no-cache -t $(IMAGE) -f docker/Dockerfile .

