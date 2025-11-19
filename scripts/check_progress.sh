#!/bin/bash

OUTPUT_FILE="changes.diff"

> "$OUTPUT_FILE"

echo "=== Modified/Tracked Changes ===" >> "$OUTPUT_FILE"
git diff HEAD >> "$OUTPUT_FILE"

echo -e "\n=== New/Untracked Files ===" >> "$OUTPUT_FILE"

git ls-files --others --exclude-standard | while read file; do
    echo -e "\n--- $file ---" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
done