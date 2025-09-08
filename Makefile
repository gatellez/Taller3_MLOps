
.PHONY: build init up down ps logs reset perms

build:
	docker compose build

perms:
	./scripts/fix_perms.sh

init: perms
	docker compose run --rm airflow-init

up:
	docker compose up -d

down:
	docker compose down

ps:
	docker compose ps

logs:
	docker compose logs -f --tail=200

reset: down
	rm -rf airflow/logs/* models/*
