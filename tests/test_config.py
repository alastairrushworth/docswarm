"""Tests for docswarm.config.Config."""

from __future__ import annotations

from docswarm.config import Config


class TestConfigDefaults:
    def test_catalog_path_default(self):
        cfg = Config()
        assert cfg.catalog_path == "docswarm_catalog.db"

    def test_data_path_default(self):
        cfg = Config()
        assert cfg.data_path == "docswarm_data/"

    def test_pdf_dir_default(self):
        cfg = Config()
        assert cfg.pdf_dir == "."

    def test_pages_dir_default(self):
        cfg = Config()
        assert cfg.pages_dir == "pages/"

    def test_wiki_url_default(self):
        cfg = Config()
        assert cfg.wiki_url == ""

    def test_wiki_api_key_default(self):
        cfg = Config()
        assert cfg.wiki_api_key == ""

    def test_wiki_output_dir_default(self):
        cfg = Config()
        assert cfg.wiki_output_dir == "wiki/"

    def test_ollama_base_url_default(self):
        cfg = Config()
        assert cfg.ollama_base_url == "http://localhost:11434"

    def test_ocr_language_default(self):
        cfg = Config()
        assert cfg.ocr_language == "eng"

    def test_chunk_size_default(self):
        cfg = Config()
        assert cfg.chunk_size == 800

    def test_chunk_overlap_default(self):
        cfg = Config()
        assert cfg.chunk_overlap == 100

    def test_model_default(self):
        cfg = Config()
        assert cfg.model == "gemma3:4b"


class TestConfigFromEnv:
    def test_reads_catalog_path(self, monkeypatch):
        monkeypatch.setenv("DOCSWARM_CATALOG_PATH", "/data/my_catalog.db")
        cfg = Config.from_env()
        assert cfg.catalog_path == "/data/my_catalog.db"

    def test_reads_data_path(self, monkeypatch):
        monkeypatch.setenv("DOCSWARM_DATA_PATH", "/data/lake/")
        cfg = Config.from_env()
        assert cfg.data_path == "/data/lake/"

    def test_reads_pdf_dir(self, monkeypatch):
        monkeypatch.setenv("DOCSWARM_PDF_DIR", "/pdfs/input")
        cfg = Config.from_env()
        assert cfg.pdf_dir == "/pdfs/input"

    def test_reads_pages_dir(self, monkeypatch):
        monkeypatch.setenv("DOCSWARM_PAGES_DIR", "/pages/output/")
        cfg = Config.from_env()
        assert cfg.pages_dir == "/pages/output/"

    def test_reads_wiki_url(self, monkeypatch):
        monkeypatch.setenv("DOCSWARM_WIKI_URL", "https://wiki.example.com")
        cfg = Config.from_env()
        assert cfg.wiki_url == "https://wiki.example.com"

    def test_reads_wiki_api_key(self, monkeypatch):
        monkeypatch.setenv("DOCSWARM_WIKI_API_KEY", "secret-key-abc123")
        cfg = Config.from_env()
        assert cfg.wiki_api_key == "secret-key-abc123"

    def test_reads_wiki_output_dir(self, monkeypatch):
        monkeypatch.setenv("DOCSWARM_WIKI_OUTPUT_DIR", "/output/wiki/")
        cfg = Config.from_env()
        assert cfg.wiki_output_dir == "/output/wiki/"

    def test_reads_ollama_base_url(self, monkeypatch):
        monkeypatch.setenv("DOCSWARM_OLLAMA_BASE_URL", "http://gpu-box:11434")
        cfg = Config.from_env()
        assert cfg.ollama_base_url == "http://gpu-box:11434"

    def test_reads_ocr_language(self, monkeypatch):
        monkeypatch.setenv("DOCSWARM_OCR_LANGUAGE", "fra")
        cfg = Config.from_env()
        assert cfg.ocr_language == "fra"

    def test_reads_chunk_size_as_int(self, monkeypatch):
        monkeypatch.setenv("DOCSWARM_CHUNK_SIZE", "1200")
        cfg = Config.from_env()
        assert cfg.chunk_size == 1200
        assert isinstance(cfg.chunk_size, int)

    def test_reads_chunk_overlap_as_int(self, monkeypatch):
        monkeypatch.setenv("DOCSWARM_CHUNK_OVERLAP", "200")
        cfg = Config.from_env()
        assert cfg.chunk_overlap == 200
        assert isinstance(cfg.chunk_overlap, int)

    def test_reads_model(self, monkeypatch):
        monkeypatch.setenv("DOCSWARM_MODEL", "llama3.2:3b")
        cfg = Config.from_env()
        assert cfg.model == "llama3.2:3b"


class TestConfigFromEnvFallsBackToDefaults:
    """When env vars are absent, Config.from_env() must use the same defaults as Config()."""

    def test_catalog_path_fallback(self, monkeypatch):
        monkeypatch.delenv("DOCSWARM_CATALOG_PATH", raising=False)
        cfg = Config.from_env()
        assert cfg.catalog_path == Config().catalog_path

    def test_chunk_size_fallback(self, monkeypatch):
        monkeypatch.delenv("DOCSWARM_CHUNK_SIZE", raising=False)
        cfg = Config.from_env()
        assert cfg.chunk_size == Config().chunk_size

    def test_model_fallback(self, monkeypatch):
        monkeypatch.delenv("DOCSWARM_MODEL", raising=False)
        cfg = Config.from_env()
        assert cfg.model == Config().model

    def test_all_defaults_match_when_no_env(self, monkeypatch):
        """from_env() with no env vars should produce the same config as Config()."""
        env_keys = [
            "DOCSWARM_CATALOG_PATH",
            "DOCSWARM_DATA_PATH",
            "DOCSWARM_PDF_DIR",
            "DOCSWARM_PAGES_DIR",
            "DOCSWARM_WIKI_URL",
            "DOCSWARM_WIKI_API_KEY",
            "DOCSWARM_WIKI_OUTPUT_DIR",
            "DOCSWARM_OLLAMA_BASE_URL",
            "DOCSWARM_OCR_LANGUAGE",
            "DOCSWARM_CHUNK_SIZE",
            "DOCSWARM_CHUNK_OVERLAP",
            "DOCSWARM_MODEL",
        ]
        for key in env_keys:
            monkeypatch.delenv(key, raising=False)
        from_env = Config.from_env()
        default = Config()
        assert from_env.catalog_path == default.catalog_path
        assert from_env.data_path == default.data_path
        assert from_env.chunk_size == default.chunk_size
        assert from_env.chunk_overlap == default.chunk_overlap
        assert from_env.model == default.model
