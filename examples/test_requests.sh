#!/usr/bin/env bash
# ==============================================================================
# Customer Churn Prediction API - Verification Script
#
# Usage:
#   ./examples/test_requests.sh
#   API_URL=http://your-remote-host:8000 ./examples/test_requests.sh
#
# Requirements:
#   - curl
#   - jq (optional, for pretty-printed JSON formatting)
# ==============================================================================

set -euo pipefail

# Configurable API host with local fallback
BASE_URL="${API_URL:-http://localhost:8000}"

echo "=================================================================="
echo " Starting API Verification: ${BASE_URL}"
echo "=================================================================="

# ------------------------------------------------------------------------------
# Test 1: Service Health & Model Availability
# ------------------------------------------------------------------------------
echo -e "\n[Step 1/3] Probing /health endpoint..."
curl -s -f "${BASE_URL}/health" | (jq . 2>/dev/null || cat)

# ------------------------------------------------------------------------------
# Test 2: Inference with High-Risk Customer Profile (Expected Churn: True)
# Short tenure (14.9 mo), Month-to-month contract, Fiber optic without tech support
# ------------------------------------------------------------------------------
echo -e "\n\n[Step 2/3] Submitting High-Risk Customer Payload (examples/high_risk_customer.json)..."
curl -s -f -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d @examples/high_risk_customer.json | (jq . 2>/dev/null || cat)

# ------------------------------------------------------------------------------
# Test 3: Inference with Low-Risk Customer Profile (Expected Churn: False)
# Long tenure (40.2 mo), Two-year contract, bundled services with tech support
# ------------------------------------------------------------------------------
echo -e "\n\n[Step 3/3] Submitting Low-Risk Customer Payload (examples/low_risk_customer.json)..."
curl -s -f -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d @examples/low_risk_customer.json | (jq . 2>/dev/null || cat)

echo -e "\n\nVerification finished successfully!"
