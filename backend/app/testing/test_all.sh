#!/bin/bash

echo "Running all tests"
for dir in */; do
    if [ -d "$dir" ] && [ "$(basename "$dir")" != ".pytest_cache" ] && [ "$(basename "$dir")" != "__pycache__" ]; then
        cd "$dir"
        echo "Running pytest in $dir"
        pytest
        cd ..
    fi
done
