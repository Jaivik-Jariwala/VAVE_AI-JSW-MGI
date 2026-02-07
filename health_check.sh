#!/bin/bash

# VAVE AI - JSW MGI Health Check Script
# This script checks the health of the application and its dependencies

# Configuration
APP_DIR="/opt/vave-ai"
SERVICE_NAME="vave-ai.service"
HEALTH_URL="http://localhost:5000"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

EXIT_CODE=0

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}VAVE AI Health Check${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Check 1: Systemd service status
echo -e "${YELLOW}[1/7] Checking systemd service...${NC}"
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo -e "${GREEN}✓ Service is running${NC}"
else
    echo -e "${RED}✗ Service is not running${NC}"
    EXIT_CODE=1
fi

# Check 2: PostgreSQL service
echo -e "${YELLOW}[2/7] Checking PostgreSQL service...${NC}"
if systemctl is-active --quiet postgresql; then
    echo -e "${GREEN}✓ PostgreSQL is running${NC}"
else
    echo -e "${RED}✗ PostgreSQL is not running${NC}"
    EXIT_CODE=1
fi

# Check 3: Database connection
echo -e "${YELLOW}[3/7] Checking database connection...${NC}"
if [ -f "$APP_DIR/.env" ]; then
    source "$APP_DIR/.env"
    if PGPASSWORD="${DB_PASSWORD:-${DB_PASS}}" psql -h "${DB_HOST:-localhost}" \
        -U "${DB_USER:-vave_user}" \
        -d "${DB_NAME:-vave_db}" \
        -c "SELECT 1;" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Database connection successful${NC}"
    else
        echo -e "${RED}✗ Database connection failed${NC}"
        EXIT_CODE=1
    fi
else
    echo -e "${YELLOW}⚠ .env file not found, skipping database check${NC}"
fi

# Check 4: Application HTTP endpoint
echo -e "${YELLOW}[4/7] Checking application HTTP endpoint...${NC}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" || echo "000")
if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "302" ] || [ "$HTTP_CODE" = "401" ]; then
    echo -e "${GREEN}✓ Application is responding (HTTP $HTTP_CODE)${NC}"
else
    echo -e "${RED}✗ Application is not responding (HTTP $HTTP_CODE)${NC}"
    EXIT_CODE=1
fi

# Check 5: Required directories
echo -e "${YELLOW}[5/7] Checking required directories...${NC}"
REQUIRED_DIRS=("$APP_DIR/model" "$APP_DIR/static" "$APP_DIR/temp" "$APP_DIR/logs")
MISSING_DIRS=0
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓ $dir exists${NC}"
    else
        echo -e "${RED}✗ $dir is missing${NC}"
        MISSING_DIRS=$((MISSING_DIRS + 1))
    fi
done
if [ $MISSING_DIRS -gt 0 ]; then
    EXIT_CODE=1
fi

# Check 6: Disk space
echo -e "${YELLOW}[6/7] Checking disk space...${NC}"
DISK_USAGE=$(df -h "$APP_DIR" | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -lt 90 ]; then
    echo -e "${GREEN}✓ Disk usage: ${DISK_USAGE}%${NC}"
else
    echo -e "${RED}✗ Disk usage is high: ${DISK_USAGE}%${NC}"
    EXIT_CODE=1
fi

# Check 7: Recent log errors
echo -e "${YELLOW}[7/7] Checking recent errors in logs...${NC}"
ERROR_COUNT=$(journalctl -u "$SERVICE_NAME" --since "5 minutes ago" --no-pager | grep -i "error\|exception\|traceback" | wc -l)
if [ "$ERROR_COUNT" -eq 0 ]; then
    echo -e "${GREEN}✓ No recent errors in logs${NC}"
else
    echo -e "${YELLOW}⚠ Found $ERROR_COUNT recent error(s) in logs${NC}"
fi

# Summary
echo -e "\n${GREEN}========================================${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Health Check: PASSED${NC}"
else
    echo -e "${RED}Health Check: FAILED${NC}"
fi
echo -e "${GREEN}========================================${NC}\n"

exit $EXIT_CODE

