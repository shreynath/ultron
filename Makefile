.PHONY: setup dev-backend dev-frontend install

setup:
	./scripts/setup.sh

install: setup

dev-backend:
	cd backend && . .venv/bin/activate && uvicorn app.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev
