.PHONY: images up down logs

images:
	docker build -t rbi-browser-session ./browser
	docker compose build api frontend

up: images
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=200
