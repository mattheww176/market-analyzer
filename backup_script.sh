#!/bin/bash
# Daily backup script for stock analysis project

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$HOME/Desktop/market_analyzer_backups"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Copy important files
cp market_analyzer.py "$BACKUP_DIR/market_analyzer_$DATE.py"
cp -r src/ "$BACKUP_DIR/src_$DATE/" 2>/dev/null || true
cp requirements.txt "$BACKUP_DIR/requirements_$DATE.txt" 2>/dev/null || true
cp README.md "$BACKUP_DIR/README_$DATE.md" 2>/dev/null || true

# Git commit if there are changes
if ! git diff-index --quiet HEAD --; then
    git add .
    git commit -m "Auto-backup: $DATE"
fi

echo "✅ Backup completed: $DATE"
echo "Files saved to: $BACKUP_DIR"

# Keep only last 10 backups (cleanup)
ls -t "$BACKUP_DIR"/market_analyzer_*.py | tail -n +11 | xargs rm -f 2>/dev/null || true
