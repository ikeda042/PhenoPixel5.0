#!/bin/bash

# ビルド
npm run build

# Webpack Dev Server の起動
npm start &

# 少し待機してから Electron を起動
sleep 5
npm run electron