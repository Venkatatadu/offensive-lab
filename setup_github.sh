#!/bin/bash
# Offensive Lab — GitHub Setup Script
# Run this in your terminal after cloning

REPO_NAME="offensive-lab"
ORG="Venkatatadu"

echo "  Offensive Lab — GitHub Setup"
echo "  ================================"

gh repo create "$ORG/$REPO_NAME" --public --description "Offensive security research tools"
git remote add origin "https://github.com/$ORG/$REPO_NAME.git"
git branch -M main
git push -u origin main

echo "Done! https://github.com/$ORG/$REPO_NAME"
