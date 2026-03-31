"""Tests for docswarm.wiki.client.WikiJSClient.

All HTTP calls are mocked via unittest.mock so no real Wiki.js instance is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from docswarm.wiki.client import WikiJSClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """WikiJSClient pointed at a fake URL with a fake API key."""
    with patch("httpx.Client"):
        c = WikiJSClient(base_url="https://wiki.example.com", api_key="test-api-key")
        yield c


def _mock_response(data: dict, status_code: int = 200):
    """Build a mock httpx.Response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = {"data": data}
    mock_resp.raise_for_status = MagicMock()  # no-op on success
    return mock_resp


def _mock_error_response(status_code: int = 500):
    """Build a mock httpx.Response that raises on raise_for_status()."""
    import httpx

    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server error", request=MagicMock(), response=mock_resp
    )
    return mock_resp


# ---------------------------------------------------------------------------
# list_pages
# ---------------------------------------------------------------------------


class TestListPages:
    def test_returns_list_of_page_dicts(self, client):
        pages_data = [
            {"id": 1, "title": "Home", "path": "home", "updatedAt": "2024-01-01"},
            {"id": 2, "title": "About", "path": "about", "updatedAt": "2024-01-02"},
        ]
        mock_resp = _mock_response({"pages": {"list": pages_data}})
        client._client.post.return_value = mock_resp

        result = client.list_pages()
        assert len(result) == 2
        assert result[0]["title"] == "Home"
        assert result[1]["path"] == "about"

    def test_returns_empty_list_when_no_pages(self, client):
        mock_resp = _mock_response({"pages": {"list": []}})
        client._client.post.return_value = mock_resp

        result = client.list_pages()
        assert result == []

    def test_sends_graphql_query(self, client):
        mock_resp = _mock_response({"pages": {"list": []}})
        client._client.post.return_value = mock_resp

        client.list_pages()
        client._client.post.assert_called_once()
        call_args = client._client.post.call_args
        assert "/graphql" in call_args[0][0]


# ---------------------------------------------------------------------------
# get_page
# ---------------------------------------------------------------------------


class TestGetPage:
    def test_returns_page_dict(self, client):
        page_data = {
            "id": 5,
            "title": "Victorian Furniture",
            "content": "# Victorian Furniture\n\nContent here.",
            "path": "history/victorian-furniture",
            "description": "An overview.",
        }
        mock_resp = _mock_response({"pages": {"single": page_data}})
        client._client.post.return_value = mock_resp

        result = client.get_page(5)
        assert result["id"] == 5
        assert result["title"] == "Victorian Furniture"
        assert result["content"] == "# Victorian Furniture\n\nContent here."

    def test_sends_page_id_as_variable(self, client):
        mock_resp = _mock_response({"pages": {"single": {"id": 7}}})
        client._client.post.return_value = mock_resp

        client.get_page(7)
        call_kwargs = client._client.post.call_args[1]
        payload = call_kwargs.get("json") or client._client.post.call_args[0][1]
        assert payload["variables"]["id"] == 7

    def test_returns_empty_dict_when_page_missing(self, client):
        mock_resp = _mock_response({"pages": {"single": {}}})
        client._client.post.return_value = mock_resp

        result = client.get_page(999)
        assert result == {}


# ---------------------------------------------------------------------------
# create_page
# ---------------------------------------------------------------------------


class TestCreatePage:
    def _success_response(self, page_id=42, path="test/path", title="Test"):
        data = {
            "pages": {
                "create": {
                    "responseResult": {
                        "succeeded": True,
                        "errorCode": None,
                        "message": None,
                        "slug": path,
                    },
                    "page": {"id": page_id, "path": path, "title": title},
                }
            }
        }
        return _mock_response(data)

    def test_returns_page_dict_on_success(self, client):
        mock_resp = self._success_response(page_id=10, path="people/test", title="Test Person")
        client._client.post.return_value = mock_resp

        result = client.create_page(
            title="Test Person",
            content="# Test Person\n\nBio here.",
            path="people/test",
            description="A test person.",
        )
        assert result["id"] == 10
        assert result["path"] == "people/test"

    def test_sends_correct_variables(self, client):
        mock_resp = self._success_response()
        client._client.post.return_value = mock_resp

        client.create_page(
            title="My Title",
            content="My content.",
            path="my/path",
            description="My description.",
        )
        call_kwargs = client._client.post.call_args[1]
        payload = call_kwargs.get("json") or client._client.post.call_args[0][1]
        variables = payload["variables"]
        assert variables["title"] == "My Title"
        assert variables["content"] == "My content."
        assert variables["path"] == "my/path"
        assert variables["description"] == "My description."

    def test_raises_runtime_error_on_failure(self, client):
        data = {
            "pages": {
                "create": {
                    "responseResult": {
                        "succeeded": False,
                        "errorCode": 404,
                        "message": "Path already in use",
                    },
                    "page": None,
                }
            }
        }
        client._client.post.return_value = _mock_response(data)

        with pytest.raises(RuntimeError, match="create_page failed"):
            client.create_page(title="X", content="X", path="x/y")


# ---------------------------------------------------------------------------
# update_page
# ---------------------------------------------------------------------------


class TestUpdatePage:
    def _success_response(self, page_id=1, path="updated/path", title="Updated"):
        data = {
            "pages": {
                "update": {
                    "responseResult": {
                        "succeeded": True,
                        "errorCode": None,
                        "message": None,
                    },
                    "page": {"id": page_id, "path": path, "title": title},
                }
            }
        }
        return _mock_response(data)

    def test_returns_updated_page_dict(self, client):
        mock_resp = self._success_response(page_id=3, path="places/london", title="London")
        client._client.post.return_value = mock_resp

        result = client.update_page(
            page_id=3,
            title="London",
            content="# London\n\nUpdated content.",
            description="The capital city.",
        )
        assert result["id"] == 3
        assert result["title"] == "London"

    def test_sends_correct_variables(self, client):
        mock_resp = self._success_response()
        client._client.post.return_value = mock_resp

        client.update_page(page_id=99, title="New Title", content="New body.")
        call_kwargs = client._client.post.call_args[1]
        payload = call_kwargs.get("json") or client._client.post.call_args[0][1]
        variables = payload["variables"]
        assert variables["id"] == 99
        assert variables["title"] == "New Title"
        assert variables["content"] == "New body."

    def test_raises_runtime_error_on_failure(self, client):
        data = {
            "pages": {
                "update": {
                    "responseResult": {
                        "succeeded": False,
                        "errorCode": 500,
                        "message": "Internal error",
                    },
                    "page": None,
                }
            }
        }
        client._client.post.return_value = _mock_response(data)

        with pytest.raises(RuntimeError, match="update_page failed"):
            client.update_page(page_id=1, title="X", content="X")


# ---------------------------------------------------------------------------
# delete_page
# ---------------------------------------------------------------------------


class TestDeletePage:
    def _success_response(self):
        data = {
            "pages": {
                "delete": {
                    "responseResult": {
                        "succeeded": True,
                        "errorCode": None,
                        "message": None,
                    }
                }
            }
        }
        return _mock_response(data)

    def test_returns_true_on_success(self, client):
        client._client.post.return_value = self._success_response()
        result = client.delete_page(42)
        assert result is True

    def test_sends_page_id_in_variables(self, client):
        client._client.post.return_value = self._success_response()
        client.delete_page(77)
        call_kwargs = client._client.post.call_args[1]
        payload = call_kwargs.get("json") or client._client.post.call_args[0][1]
        assert payload["variables"]["id"] == 77

    def test_raises_runtime_error_on_failure(self, client):
        data = {
            "pages": {
                "delete": {
                    "responseResult": {
                        "succeeded": False,
                        "errorCode": 404,
                        "message": "Page not found",
                    }
                }
            }
        }
        client._client.post.return_value = _mock_response(data)

        with pytest.raises(RuntimeError, match="delete_page failed"):
            client.delete_page(999)


# ---------------------------------------------------------------------------
# search_pages
# ---------------------------------------------------------------------------


class TestSearchPages:
    def test_returns_list_of_results(self, client):
        results_data = [
            {
                "id": 1,
                "title": "Chippendale",
                "path": "people/chippendale",
                "description": "Furniture maker",
            },
            {
                "id": 2,
                "title": "Sheraton",
                "path": "people/sheraton",
                "description": "Another maker",
            },
        ]
        mock_resp = _mock_response({"pages": {"search": {"results": results_data}}})
        client._client.post.return_value = mock_resp

        result = client.search_pages("furniture maker")
        assert len(result) == 2
        assert result[0]["title"] == "Chippendale"

    def test_returns_empty_list_when_no_matches(self, client):
        mock_resp = _mock_response({"pages": {"search": {"results": []}}})
        client._client.post.return_value = mock_resp

        result = client.search_pages("xyznotfound")
        assert result == []

    def test_sends_query_variable(self, client):
        mock_resp = _mock_response({"pages": {"search": {"results": []}}})
        client._client.post.return_value = mock_resp

        client.search_pages("my search term")
        call_kwargs = client._client.post.call_args[1]
        payload = call_kwargs.get("json") or client._client.post.call_args[0][1]
        assert payload["variables"]["query"] == "my search term"


# ---------------------------------------------------------------------------
# _graphql error handling
# ---------------------------------------------------------------------------


class TestGraphqlErrorHandling:
    def test_raises_http_status_error_on_non_2xx(self, client):
        import httpx

        client._client.post.return_value = _mock_error_response(503)

        with pytest.raises(httpx.HTTPStatusError):
            client._graphql("query { pages { list { id } } }")

    def test_raises_runtime_error_on_graphql_errors(self, client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "errors": [
                {"message": "Unauthorised"},
                {"message": "Token expired"},
            ]
        }
        client._client.post.return_value = mock_resp

        with pytest.raises(RuntimeError, match="GraphQL errors"):
            client._graphql("query { pages { list { id } } }")

    def test_error_message_includes_all_graphql_errors(self, client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "errors": [
                {"message": "First error"},
                {"message": "Second error"},
            ]
        }
        client._client.post.return_value = mock_resp

        with pytest.raises(RuntimeError) as exc_info:
            client._graphql("query { pages { list { id } } }")

        assert "First error" in str(exc_info.value)
        assert "Second error" in str(exc_info.value)

    def test_returns_data_on_success(self, client):
        mock_resp = _mock_response({"pages": {"list": [{"id": 1}]}})
        client._client.post.return_value = mock_resp

        data = client._graphql("query { pages { list { id } } }")
        assert "pages" in data


# ---------------------------------------------------------------------------
# Client lifecycle
# ---------------------------------------------------------------------------


class TestWikiClientLifecycle:
    def test_close_calls_http_client_close(self, client):
        client.close()
        client._client.close.assert_called_once()

    def test_context_manager_closes_on_exit(self):
        with patch("httpx.Client"):
            with WikiJSClient("https://example.com", "key") as c:
                mock_close = MagicMock()
                c._client.close = mock_close
            mock_close.assert_called_once()

    def test_base_url_strips_trailing_slash(self):
        with patch("httpx.Client"):
            c = WikiJSClient("https://wiki.example.com/", "key")
            assert c.base_url == "https://wiki.example.com"
