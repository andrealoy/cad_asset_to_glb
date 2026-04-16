APP_NAME     = cad-asset-to-glb
AWS_REGION   = eu-west-3
AWS_ACCOUNT  = $(shell aws sts get-caller-identity --query Account --output text)
ECR_REPO     = $(AWS_ACCOUNT).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP_NAME)

.PHONY: clean-views ecr-create docker-build docker-push deploy

clean-views:
	rm -rf views/*

# 1. Créer le repo ECR (une seule fois)
ecr-create:
	aws ecr create-repository --repository-name $(APP_NAME) --region $(AWS_REGION)

# 2. Build l'image Docker
docker-build:
	docker build --platform linux/amd64 -t $(APP_NAME) .

# 3. Push vers ECR
docker-push:
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ECR_REPO)
	docker tag $(APP_NAME):latest $(ECR_REPO):latest
	docker push $(ECR_REPO):latest

# Tout-en-un : build + push
deploy: docker-build docker-push
	@echo "✓ Image pushée : $(ECR_REPO):latest"
	@echo "→ Va sur https://$(AWS_REGION).console.aws.amazon.com/apprunner pour créer le service"
