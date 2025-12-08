#!/bin/bash

# Script to download files from specific commit 0f6de17
# and handle folder name changes (removed leading numbers)

COMMIT_HASH="0f6de17d45d7fc4252d2c0d9c2ea3ed66715b8b9"
REPO_URL="https://github.com/UCBoulder/optimizedsampling"
TEMP_DIR="temp_commit_0f6de17"

# Create temporary directory
mkdir -p "$TEMP_DIR"

echo "Downloading files from commit $COMMIT_HASH..."

# Method 1: Clone the entire repo at that commit (recommended for complete accuracy)
echo "Cloning repository at specific commit..."
git clone "$REPO_URL" "$TEMP_DIR/repo_temp"
cd "$TEMP_DIR/repo_temp"
git checkout "$COMMIT_HASH"
cd ../..

# Move files out of git repo to remove git tracking
echo "Copying files to non-tracked directory..."
cp -r "$TEMP_DIR/repo_temp/"* "$TEMP_DIR/"
cp -r "$TEMP_DIR/repo_temp/".* "$TEMP_DIR/" 2>/dev/null || true

# Remove the git repository to ensure no tracking
rm -rf "$TEMP_DIR/repo_temp/.git"
rm -rf "$TEMP_DIR/.git"

echo "Files downloaded to $TEMP_DIR/"
echo ""
echo "Folder mapping (if folders had leading numbers removed):"
echo "Example mappings you may need:"
echo "  Old: 01_folder_name/ → New: folder_name/"
echo "  Old: 02_another_folder/ → New: another_folder/"
echo ""
echo "To map to your current structure, you can:"
echo "1. Manually identify which old folders map to your new folders"
echo "2. Copy specific files: cp $TEMP_DIR/old_path/file.txt new_path/file.txt"
echo ""
echo "Files are NOT tracked by git (local only)."
echo ""

# Add temp directory to .gitignore if not already there
if ! grep -q "^$TEMP_DIR/\$" .gitignore 2>/dev/null; then
    echo "$TEMP_DIR/" >> .gitignore
    echo "Added $TEMP_DIR/ to .gitignore"
fi

# List the directory structure
echo "Directory structure downloaded:"
find "$TEMP_DIR" -type f | head -20
echo "..."
echo ""
echo "Total files: $(find "$TEMP_DIR" -type f | wc -l)"
