"""Managed collection stays opt-in and accepts an API origin, never a sharing link."""

import pytest

from app.core.config import Settings


def account(**extra):
    return Settings(_env_file=None, nextcloud_backend="account", nextcloud_base_url="https://cloud.example",
                    nextcloud_username="service", nextcloud_app_password="test-only-password", **extra)


def test_workstation_collection_is_disabled_by_default_and_needs_no_endpoint_in_existing_installations():
    assert Settings(_env_file=None).device_collection_base_url == ""
    assert account().device_collection_base_url == ""


@pytest.mark.parametrize("url", ["https://support.example", "https://support.example/ai-support/"])
def test_administrator_can_configure_an_https_api_origin_or_proxy_prefix(url):
    assert account(device_collection_base_url=url).device_collection_base_url == url


@pytest.mark.parametrize("url", [
    "http://support.example", "https://user:pass@support.example", "https://support.example?share=token",
    "https://support.example/#token=123", "https://support.example/a/../b", "https://support.example/a\\b",
    "https://support.example/a b", "https://support.example/\n",
])
def test_invalid_or_shared_document_urls_cannot_enable_the_managed_callback(url):
    with pytest.raises(ValueError):
        account(device_collection_base_url=url)


def test_webhook_or_mock_backend_cannot_accidentally_enable_the_account_collection_feature():
    with pytest.raises(ValueError):
        Settings(_env_file=None, device_collection_base_url="https://support.example")
