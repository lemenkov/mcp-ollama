#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: 2026 Peter Lemenkov <lemenkov@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Install the Ollama MCP server on Sulaco.
# Run as root:  sudo bash ~/ollama-mcp/install-mcp-ollama.sh
#
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
    echo "This script must run as root. Use: sudo bash $0" >&2
    exit 1
fi

SRC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DST_DIR="/opt/mcp-servers/mcp-ollama"
SYSCONFIG="/etc/sysconfig/mcp-ollama"
BACKENDS="/etc/sysconfig/mcp-ollama.json"
UNIT="/etc/systemd/system/mcp-ollama.service"
MCP_CONF="/etc/nginx/conf.d/mcp.conf"

echo "==> installing code into ${DST_DIR}"
install -d "${DST_DIR}"
rm -rf "${DST_DIR}/mcp_ollama" "${DST_DIR}/examples"
cp -a "${SRC_DIR}/mcp_ollama" "${DST_DIR}/"
cp -a "${SRC_DIR}/pyproject.toml" "${DST_DIR}/"
cp -a "${SRC_DIR}/README.md" "${DST_DIR}/"
for extra in LICENSE NOTICE; do
    if [[ -f "${SRC_DIR}/${extra}" ]]; then
        cp -a "${SRC_DIR}/${extra}" "${DST_DIR}/"
    fi
done
if [[ -d "${SRC_DIR}/examples" ]]; then
    cp -a "${SRC_DIR}/examples" "${DST_DIR}/"
fi
chown -R root:root "${DST_DIR}/mcp_ollama" "${DST_DIR}/pyproject.toml" \
    "${DST_DIR}/README.md" 2>/dev/null || true
for extra in LICENSE NOTICE; do
    if [[ -f "${DST_DIR}/${extra}" ]]; then
        chown root:root "${DST_DIR}/${extra}" 2>/dev/null || true
    fi
done
if [[ -d "${DST_DIR}/examples" ]]; then
    chown -R root:root "${DST_DIR}/examples" 2>/dev/null || true
fi

if [[ ! -x "${DST_DIR}/venv/bin/python" ]]; then
    echo "==> creating virtualenv"
    python3 -m venv "${DST_DIR}/venv"
fi

echo "==> installing Python dependencies"
"${DST_DIR}/venv/bin/pip" install --disable-pip-version-check --upgrade pip
"${DST_DIR}/venv/bin/pip" install --disable-pip-version-check "${DST_DIR}"

echo "==> installing configuration"
if [[ -f "${SYSCONFIG}" ]]; then
    echo "keeping existing ${SYSCONFIG}"
else
    install -m 0644 "${SRC_DIR}/mcp-ollama.sysconfig" "${SYSCONFIG}"
fi

if [[ -f "${BACKENDS}" ]]; then
    echo "keeping existing ${BACKENDS}"
else
    install -m 0600 "${SRC_DIR}/mcp-ollama.json" "${BACKENDS}"
fi

# The listener and the nginx proxy must agree on the port.
MCP_PORT="$(. "${SYSCONFIG}" >/dev/null 2>&1; printf '%s' "${PORT:-8815}")"
MCP_PORT="${MCP_PORT//[[:space:]]/}"
MCP_PORT="${MCP_PORT:-8815}"

# Fail before restarting anything if the backend JSON is structurally invalid.
"${DST_DIR}/venv/bin/python" -c \
    'from mcp_ollama.config import load_config; load_config("/etc/sysconfig/mcp-ollama.json")' \
    >/dev/null

echo "==> installing systemd unit"
if [[ -f "${UNIT}" ]]; then
    cp -a "${UNIT}" "${UNIT}.bak-$(date +%Y%m%d-%H%M%S)"
fi
install -m 0644 "${SRC_DIR}/mcp-ollama.service" "${UNIT}"
systemctl daemon-reload
systemctl enable mcp-ollama.service
if ! systemctl restart mcp-ollama.service; then
    echo "mcp-ollama failed to start; recent journal follows:" >&2
    journalctl -u mcp-ollama.service -n 60 --no-pager >&2 || true
    exit 1
fi

echo "==> configuring nginx"
if command -v nginx >/dev/null 2>&1 && [[ -f "${MCP_CONF}" ]]; then
    cp -a "${MCP_CONF}" "${MCP_CONF}.bak-$(date +%Y%m%d-%H%M%S)"
    "${DST_DIR}/venv/bin/python" - "${MCP_CONF}" "${MCP_PORT}" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
port = sys.argv[2]
text = path.read_text()

if "/mcp/ollama" in text:
    match = re.search(
        r"location \^~ /mcp/ollama \{.*?proxy_pass http://127\.0\.0\.1:(\d+);",
        text,
        re.S,
    )
    if match is None:
        raise SystemExit(
            "existing /mcp/ollama location found, but its proxy_pass could not be read"
        )
    if match.group(1) != port:
        raise SystemExit(
            f"existing /mcp/ollama proxies to {match.group(1)}, but PORT={port}; "
            "update mcp.conf or the sysconfig"
        )
    print("nginx location for /mcp/ollama already present")
    raise SystemExit(0)

anchor = "    location ^~ /mcp/nvd {"
block = f"""    location ^~ /mcp/ollama {
        if ($http_accept ~* \"text/html\") {
            rewrite ^ /_mcp_hello last;
        }
        rewrite ^/mcp/ollama(.*)$ /mcp$1 break;
        include snippets/mcp-proxy-headers.conf;
        proxy_pass http://127.0.0.1:{port};
    }

"""
if anchor not in text:
    raise SystemExit("anchor not found in mcp.conf; refusing to edit")

path.write_text(text.replace(anchor, block + anchor, 1))
print(f"inserted /mcp/ollama location on port {port}")
PY
    if nginx -t; then
        systemctl reload nginx
    else
        echo "nginx configuration test failed; restoring backup" >&2
        latest="$(ls -1t "${MCP_CONF}".bak-* 2>/dev/null | head -1 || true)"
        if [[ -n "${latest}" ]]; then
            cp -a "${latest}" "${MCP_CONF}"
        fi
        exit 1
    fi
else
    echo "warning: nginx or ${MCP_CONF} not found; add the /mcp/ollama proxy manually" >&2
fi

echo
echo "==> service status"
systemctl --no-pager --full status mcp-ollama.service || true

echo
echo "==> listener"
ss -ltn 2>/dev/null | grep -E ":${MCP_PORT}\b" || true

echo
echo "==> recent journal"
journalctl -u mcp-ollama.service -n 25 --no-pager || true

echo
echo "MCP endpoint: https://lemenkov.duckdns.org/mcp/ollama"
echo "It uses the same Basic auth as the other /mcp routes."
