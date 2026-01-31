#!/usr/bin/env bash
set -euo pipefail

# Crawl4AI post-install helper
# Usage: run from project root: ./scripts/crawl4ai_postinstall.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MARKER_FILE="$ROOT_DIR/.crawl4ai_installed"
echo "Project root: $ROOT_DIR"

# Exit early if postinstall already ran
if [ -f "$MARKER_FILE" ]; then
  echo "Post-install already completed (marker: $MARKER_FILE). Skipping."
  exit 0
fi

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

if ! command_exists crawl4ai-setup; then
  echo "crawl4ai CLI not found. Installing crawl4ai via pip..."
  python -m pip install "crawl4ai"
else
  echo "crawl4ai CLI already available"
fi

echo "Running crawl4ai-setup (may download Playwright browsers and models)"
crawl4ai-setup || { echo "crawl4ai-setup failed"; exit 1; }

echo "Attempting to download recommended models (may be large)."
if command_exists crawl4ai-download-models; then
  crawl4ai-download-models || echo "crawl4ai-download-models failed (continuing)"
else
  echo "crawl4ai-download-models command not available for this installation; skip."
fi

echo "Verifying installation with crawl4ai-doctor (best-effort)"
if command_exists crawl4ai-doctor; then
  crawl4ai-doctor || echo "crawl4ai-doctor reported issues"
else
  echo "crawl4ai-doctor not available for this installation; skip."
fi

echo "Crawl4AI post-install completed. Creating marker: $MARKER_FILE"
touch "$MARKER_FILE"
echo "Marker created. Future runs will skip post-install."

