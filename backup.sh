#!/bin/bash

# VAVE AI - JSW MGI Backup Script
# This script backs up the database and important files

set -e

# Configuration
APP_DIR="/opt/vave-ai"
APP_USER="vave"
BACKUP_DIR="$APP_DIR/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="vave_backup_$TIMESTAMP"
BACKUP_PATH="$BACKUP_DIR/$BACKUP_NAME"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting backup...${NC}"

# Create backup directory
mkdir -p "$BACKUP_PATH"

# Load environment variables
if [ -f "$APP_DIR/.env" ]; then
    source "$APP_DIR/.env"
else
    echo -e "${RED}Error: .env file not found${NC}"
    exit 1
fi

# Backup PostgreSQL database
echo -e "${YELLOW}Backing up PostgreSQL database...${NC}"
PGPASSWORD="${DB_PASSWORD:-${DB_PASS}}" pg_dump -h "${DB_HOST:-localhost}" \
    -U "${DB_USER:-vave_user}" \
    -d "${DB_NAME:-vave_db}" \
    -F c \
    -f "$BACKUP_PATH/database.dump"

# Backup SQLite users database
if [ -f "$APP_DIR/users.db" ]; then
    echo -e "${YELLOW}Backing up SQLite users database...${NC}"
    cp "$APP_DIR/users.db" "$BACKUP_PATH/users.db"
fi

# Backup important directories
echo -e "${YELLOW}Backing up application files...${NC}"
tar -czf "$BACKUP_PATH/model.tar.gz" -C "$APP_DIR" model/ 2>/dev/null || echo "Model directory not found or empty"
tar -czf "$BACKUP_PATH/images.tar.gz" -C "$APP_DIR" images/ 2>/dev/null || echo "Images directory not found or empty"
tar -czf "$BACKUP_PATH/data_lake.tar.gz" -C "$APP_DIR" data_lake/ 2>/dev/null || echo "Data lake directory not found or empty"

# Backup configuration files
echo -e "${YELLOW}Backing up configuration files...${NC}"
cp "$APP_DIR/.env" "$BACKUP_PATH/.env" 2>/dev/null || echo ".env file not found"
cp "$APP_DIR/requirements.txt" "$BACKUP_PATH/requirements.txt" 2>/dev/null || true

# Create backup info file
cat > "$BACKUP_PATH/backup_info.txt" <<EOF
Backup Date: $(date)
Application: VAVE AI - JSW MGI
Database: ${DB_NAME:-vave_db}
Host: ${DB_HOST:-localhost}
Backup Type: Full
EOF

# Create compressed archive
echo -e "${YELLOW}Creating compressed archive...${NC}"
cd "$BACKUP_DIR"
tar -czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME"
rm -rf "$BACKUP_NAME"

# Set ownership
chown -R "$APP_USER:$APP_USER" "$BACKUP_DIR"

# Cleanup old backups (keep last 7 days)
echo -e "${YELLOW}Cleaning up old backups (keeping last 7 days)...${NC}"
find "$BACKUP_DIR" -name "vave_backup_*.tar.gz" -mtime +7 -delete

echo -e "${GREEN}Backup completed: ${BACKUP_NAME}.tar.gz${NC}"
echo -e "${GREEN}Backup size: $(du -h "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" | cut -f1)${NC}"

