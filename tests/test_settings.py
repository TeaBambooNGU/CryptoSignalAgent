"""Settings 配置加载测试。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config.settings import Settings


class SettingsTestCase(unittest.TestCase):
    def test_load_mcp_servers_from_claude_style_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / ".mcp.json"
            config_path.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "coingecko": {
                                "type": "http",
                                "url": "https://mcp.api.coingecko.com/mcp",
                            },
                            "crypto_news": {
                                "type": "stdio",
                                "command": "uv",
                                "args": ["run", "crypto-news-mcp"],
                                "env": {"CRYPTOPANIC_AUTH_TOKEN": "${CRYPTOPANIC_AUTH_TOKEN}"},
                            },
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "MCP_CONFIG_PATH": str(config_path),
                    "CRYPTOPANIC_AUTH_TOKEN": "token-for-test",
                },
                clear=False,
            ):
                settings = Settings.from_env(env_file=str(Path(tmp) / "not-exist.env"))

        self.assertIn("coingecko", settings.mcp_servers)
        self.assertIn("crypto_news", settings.mcp_servers)
        self.assertEqual(
            settings.mcp_servers["crypto_news"]["env"]["CRYPTOPANIC_AUTH_TOKEN"],
            "token-for-test",
        )

    def test_missing_mcp_config_returns_empty_servers(self) -> None:
        with patch.dict(
            os.environ,
            {"MCP_CONFIG_PATH": "/tmp/crypto_signal_agent_missing_mcp_config.json"},
            clear=False,
        ):
            settings = Settings.from_env(env_file="/tmp/crypto_signal_agent_not_exist.env")
        self.assertEqual(settings.mcp_servers, {})


if __name__ == "__main__":
    unittest.main()
