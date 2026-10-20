#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-v2_dsar_initial}"
BACKUP_DIR="$(cd "$(dirname "$0")" && pwd)/${VERSION}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -d "${BACKUP_DIR}" ]; then
    echo "Error: Backup version '${VERSION}' not found in $(dirname "$0")."
    echo "Available versions:"
    ls -1 "$(dirname "$0")" | grep -v "rollback.sh"
    exit 1
fi

cp "${BACKUP_DIR}/coded_tools/tools/detect_pii.py" "${PROJECT_ROOT}/coded_tools/tools/detect_pii.py"
cp "${BACKUP_DIR}/coded_tools/tools/redact_document.py" "${PROJECT_ROOT}/coded_tools/tools/redact_document.py"
cp "${BACKUP_DIR}/registries/generated/hitl_document_redaction_obfuscation_network.hocon" "${PROJECT_ROOT}/registries/generated/hitl_document_redaction_obfuscation_network.hocon"
cp "${BACKUP_DIR}/registries/generated/redaction_pipeline.hocon" "${PROJECT_ROOT}/registries/generated/redaction_pipeline.hocon"

echo "Successfully rolled back to version: ${VERSION}"

