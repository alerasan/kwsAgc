#!/usr/bin/env bash
apt update
apt -y upgrade
apt install -y python3
apt install -y python3-pip

pip install -r requirements.txt