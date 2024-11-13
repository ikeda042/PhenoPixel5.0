#!/bin/bash

# Root directory of the testing folder
root_dir="./testing"

# Find all test_---.py files in subdirectories and run pytest for each
find "$root_dir" -type f -name "test_*.py" | while read test_file; do
    echo "Running pytest for $test_file"
    pytest "$test_file"
done
