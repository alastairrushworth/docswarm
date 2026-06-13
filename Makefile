.PHONY: run down setup test report clean help

PYTHON ?= $(shell command -v python3 || command -v python)

help:
	@echo "Targets:"
	@echo "  make setup     # one-time: create RunPod network volume + install deps"
	@echo "  make run       # provision GPU pod, run loop, tear down on exit"
	@echo "  make down      # destroy a pod whose ID we recorded (recovery)"
	@echo "  make test      # run module against test set (user-only, local)"
	@echo "  make report    # print latest round_history table"
	@echo "  make clean     # clear inbox/feedback/cache"

setup:
	$(PYTHON) orchestration/launch.py setup

run:
	$(PYTHON) orchestration/launch.py up

down:
	$(PYTHON) orchestration/launch.py down

test:
	$(PYTHON) scripts/run_test.py

report:
	$(PYTHON) scripts/report.py

clean:
	rm -rf judge/inbox/*.json judge/feedback/*.json
	rm -rf .cache/pdf_to_json
