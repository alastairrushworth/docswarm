.PHONY: run-remote run-local run-local-fast build-snapshot test report clean help

help:
	@echo "Targets:"
	@echo "  make run-remote      # provision H100, run loop, tear down"
	@echo "  make run-local       # local docker-compose, smoke-test plumbing"
	@echo "  make run-local-fast  # local, single PDF, stub model, <60s"
	@echo "  make build-snapshot  # rebuild the DigitalOcean snapshot"
	@echo "  make test            # run module against test set (user-only)"
	@echo "  make report          # print latest round_history table"

run-remote:
	bash orchestration/run.sh remote

run-local:
	@echo "LOCAL MODE: plumbing test only. Output quality is not representative"
	@echo "of remote H100 runs because Metal/CPU and CUDA produce different model"
	@echo "outputs. Use this to verify containers come up and data flows, not to"
	@echo "evaluate translation quality."
	bash orchestration/run.sh local

run-local-fast:
	DOCSWARM_MODE=local python scripts/run_local_fast.py

build-snapshot:
	python orchestration/provision.py build-snapshot

test:
	python scripts/run_test.py

report:
	python scripts/report.py

clean:
	rm -rf judge/inbox/*.json judge/feedback/*.json
	rm -rf .cache/pdf_to_json
