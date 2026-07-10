#!/usr/bin/env bash
# Runs when a Claude Code cloud session is created. Installs the system
# dependency (espeak-ng, for IPA) and fetches the embeddable fonts.
set -e
sudo apt-get update -y && sudo apt-get install -y espeak-ng
pip install -r requirements.txt
bash fonts/fetch_fonts.sh || true
