APP_NAME     = cad-asset-to-glb
AWS_REGION   = eu-west-3
AWS_ACCOUNT  = $(shell aws sts get-caller-identity --query Account --output text)
ECR_REPO     = $(AWS_ACCOUNT).dkr.ecr.$(AWS_REGION).amazonaws.com/$(APP_NAME)

.PHONY: clean-views ecr-create docker-build docker-push deploy run run-native

clean-views:
	rm -rf views/*

# 1. Créer le repo ECR (une seule fois)
ecr-create:
	aws ecr create-repository --repository-name $(APP_NAME) --region $(AWS_REGION)

# 2. Build l'image Docker pour AWS (linux/amd64) — utilisé pour le déploiement.
#    Note : cascadio ne publie PAS de wheel linux/arm64, donc on ne peut pas
#    construire en arm64 natif dans Docker. Pour du dev local rapide, utilise
#    `make run-native` qui lance l'app dans le venv Python du Mac (cascadio a
#    un wheel macOS arm64 natif → pas d'émulation).
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

# Lance l'image AWS (amd64) dans Docker — émulée sur Apple Silicon (lent).
run:
	docker run -p 8080:8080 $(APP_NAME):latest

# Lance l'app en natif dans le venv local (rapide sur Apple Silicon, pas de Docker).
# Nécessite : python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
#
# IMPORTANT : 1 worker + plusieurs threads (gthread).
# Le registre des jobs (_JOBS) et la redirection de stderr (os.dup2) vivent
# en mémoire d'un seul processus. Avec plusieurs workers forkés, un POST
# /convert et son GET /convert/stream/<id> peuvent atterrir sur des
# processus différents → 404 + "SSE connection lost". Avec gthread, tous
# les requests partagent la même mémoire.
run-native:
	. .venv/bin/activate && gunicorn --bind 0.0.0.0:8080 \
		--worker-class gthread --workers 1 --threads 8 \
		--timeout 600 app:app