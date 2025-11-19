#!/bin/bash
# Simple dependency installer for Quantsploit

echo "====================================="
echo "Installing Quantsploit Dependencies"
echo "====================================="

# Install dependencies
pip3 install --user -r requirements.txt

echo ""
echo "====================================="
echo "Installation Complete!"
echo "====================================="
echo ""
echo "Run Quantsploit with:"
echo "  python3 quantsploit.py"
echo ""
