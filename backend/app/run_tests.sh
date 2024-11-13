#!/bin/bash

# Root directory of the testing folder
root_dir="./testing"

# Find all test_---.py files in subdirectories and run pytest for each
find "$root_dir" -type f -name "test_*.py" | while read test_file; do
    test_dir=$(dirname "$test_file")
    echo "Running pytest in $test_dir"
    cd "$test_dir" || exit 1
    pytest "$(basename "$test_file")"
    cd - > /dev/null
done
