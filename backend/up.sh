#!/bin/bash

# ネットワークが存在するか確認し、存在しない場合は作成
network_name="webnet"

if ! docker network ls | grep -q $network_name; then
    echo "Creating network $network_name..."
    docker network create $network_name
else
    echo "Network $network_name already exists."
fi

# Traefikサービスを起動
echo "Starting Traefik service..."
docker-compose -f docker-compose.yml up -d traefik

# プロジェクトサービスをビルドして起動
echo "Starting project services..."
docker-compose -f docker-compose.yml up --build -d web

echo "All services are up and running."