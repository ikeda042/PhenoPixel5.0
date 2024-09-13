#!/bin/bash

# Define the template and output file names
TEMPLATE_FILE="env_template.env"
OUTPUT_FILE=".env"

# Check if the template file exists
if [ ! -f "$TEMPLATE_FILE" ]; then
  echo "Template file $TEMPLATE_FILE not found!"
  exit 1
fi

# Copy the template to the .env file
cp "$TEMPLATE_FILE" "$OUTPUT_FILE"

# Replace HOST_NAME= with HOST_NAME=localhost in the .env file
sed -i 's/HOST_NAME=/HOST_NAME=localhost/' "$OUTPUT_FILE"

# Output success message
echo ".env file has been successfully generated with HOST_NAME=localhost."