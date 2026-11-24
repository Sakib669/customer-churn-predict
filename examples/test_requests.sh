#!/usr/bin/env bash
# Quick verification script for Customer Churn Prediction API

BASE_URL="${API_URL:-http://localhost:8000}"

echo "============================================="
echo "Testing API at: ${BASE_URL}"
echo "============================================="

echo -e "\n1. Checking Health Endpoint..."
curl -s "${BASE_URL}/health" | jq . || curl -s "${BASE_URL}/health"

echo -e "\n\n2. Predicting High-Risk Customer..."
curl -s -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d @examples/high_risk_customer.json | jq . || curl -s -X POST "${BASE_URL}/predict" -H "Content-Type: application/json" -d @examples/high_risk_customer.json

echo -e "\n\n3. Predicting Low-Risk Customer..."
curl -s -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d @examples/low_risk_customer.json | jq . || curl -s -X POST "${BASE_URL}/predict" -H "Content-Type: application/json" -d @examples/low_risk_customer.json

echo -e "\n\nDone."
