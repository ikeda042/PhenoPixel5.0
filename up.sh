#!/usr/bin/env bash

cd "$(dirname "$0")"

cd backend/app

python main.py &

cd ../../frontend
npm start

