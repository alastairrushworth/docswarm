.PHONY: run down snapshot test report clean help

help:
	@echo "Targets:"
	@echo "  make run       # provision H200 droplet, run loop, tear down on exit"
	@echo "  make down      # destroy a droplet whose ID we recorded (recovery)"
	@echo "  make snapshot  # walk through snapshot creation"
	@echo "  make test      # run module against test set (user-only, local)"
	@echo "  make report    # print latest round_history table"
	@echo "  make clean     # clear inbox/feedback/cache"

run:
	python orchestration/launch.py up

down:
	python orchestration/launch.py down

snapshot:
	python orchestration/launch.py snapshot

test:
	python scripts/run_test.py

report:
	python scripts/report.py

clean:
	rm -rf judge/inbox/*.json judge/feedback/*.json
	rm -rf .cache/pdf_to_json
