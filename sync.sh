#!/usr/bin/env bash


rsync -avP \
   --exclude='lightning_logs' \
   --exclude='venv' \
   --exclude='__pycache__' \
   --exclude='frontend' \
   --exclude='.idea' \
   . $1:./projects/cloud-faq/
