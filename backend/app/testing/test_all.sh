#!/bin/bash

echo "Running all tests"
echo "Make sure server_dev.sh has already been executed for the payment related tests"

for dir in */; do
    if [ -d "$dir" ] && [ "$(basename "$dir")" != ".pytest_cache" ] && [ "$(basename "$dir")" != "__pycache__" ]; then
        cd "$dir"
        echo "Running pytest in $dir"
        pytest
        cd ..
    fi
done
