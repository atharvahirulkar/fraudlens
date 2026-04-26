#!/usr/bin/env bash
# One-time AWS infrastructure bootstrap for FraudLens.
# Run this once from your local machine before the first deploy.
#
# Prerequisites:
#   brew install awscli
#   aws configure   (or set AWS_PROFILE)
#
# Usage:
#   chmod +x infra/setup.sh
#   GITHUB_ORG=your-github-username GITHUB_REPO=fraudlens ./infra/setup.sh

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="fraudlens"
S3_BUCKET="fraudlens-artifacts-${ACCOUNT_ID}"
APP_RUNNER_SERVICE="fraudlens-inference"
GITHUB_ORG="${GITHUB_ORG:?Set GITHUB_ORG to your GitHub username/org}"
GITHUB_REPO="${GITHUB_REPO:-fraudlens}"

echo "======================================================"
echo " FraudLens AWS Bootstrap"
echo " Account : $ACCOUNT_ID"
echo " Region  : $REGION"
echo " S3      : $S3_BUCKET"
echo "======================================================"

# - 1. ECR repository 
echo ""
echo "→ Creating ECR repository: $ECR_REPO"
aws ecr create-repository \
  --repository-name "$ECR_REPO" \
  --region "$REGION" \
  --image-scanning-configuration scanOnPush=true \
  --output json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('  URI:', d['repository']['repositoryUri'])" \
  || echo "  (already exists)"

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"

# - 2. S3 bucket for model artifacts
echo ""
echo "→ Creating S3 bucket: $S3_BUCKET"
if [ "$REGION" = "us-east-1" ]; then
  aws s3api create-bucket --bucket "$S3_BUCKET" --region "$REGION" 2>/dev/null || echo "  (already exists)"
else
  aws s3api create-bucket --bucket "$S3_BUCKET" --region "$REGION" \
    --create-bucket-configuration LocationConstraint="$REGION" 2>/dev/null || echo "  (already exists)"
fi
aws s3api put-bucket-versioning \
  --bucket "$S3_BUCKET" \
  --versioning-configuration Status=Enabled

echo ""
echo "→ Uploading model artifacts to S3"
aws s3 sync mlruns/ "s3://${S3_BUCKET}/mlruns/" --region "$REGION"
echo "  Uploaded: s3://${S3_BUCKET}/mlruns/"

# - 3. IAM role: App Runner → ECR
echo ""
echo "→ Creating IAM role: AppRunnerECRAccessRole"
aws iam create-role \
  --role-name AppRunnerECRAccessRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "build.apprunner.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }' 2>/dev/null || echo "  (already exists)"

aws iam attach-role-policy \
  --role-name AppRunnerECRAccessRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess \
  2>/dev/null || true

ECR_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/AppRunnerECRAccessRole"

# - 4. IAM role: GitHub Actions OIDC
echo ""
echo "→ Creating OIDC provider for GitHub Actions"
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 \
  2>/dev/null || echo "  (already exists)"

OIDC_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"

echo ""
echo "→ Creating IAM role: GitHubActionsRole"
aws iam create-role \
  --role-name GitHubActionsRole \
  --assume-role-policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [{
      \"Effect\": \"Allow\",
      \"Principal\": {\"Federated\": \"${OIDC_ARN}\"},
      \"Action\": \"sts:AssumeRoleWithWebIdentity\",
      \"Condition\": {
        \"StringEquals\": {
          \"token.actions.githubusercontent.com:aud\": \"sts.amazonaws.com\"
        },
        \"StringLike\": {
          \"token.actions.githubusercontent.com:sub\": \"repo:${GITHUB_ORG}/${GITHUB_REPO}:*\"
        }
      }
    }]
  }" 2>/dev/null || echo "  (already exists)"

# Inline policy: ECR push + S3 sync + App Runner describe
aws iam put-role-policy \
  --role-name GitHubActionsRole \
  --policy-name FraudLensDeployPolicy \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {
        \"Sid\": \"ECR\",
        \"Effect\": \"Allow\",
        \"Action\": [
          \"ecr:GetAuthorizationToken\",
          \"ecr:BatchCheckLayerAvailability\",
          \"ecr:PutImage\",
          \"ecr:InitiateLayerUpload\",
          \"ecr:UploadLayerPart\",
          \"ecr:CompleteLayerUpload\",
          \"ecr:BatchGetImage\"
        ],
        \"Resource\": \"*\"
      },
      {
        \"Sid\": \"S3ModelArtifacts\",
        \"Effect\": \"Allow\",
        \"Action\": [\"s3:GetObject\", \"s3:ListBucket\"],
        \"Resource\": [
          \"arn:aws:s3:::${S3_BUCKET}\",
          \"arn:aws:s3:::${S3_BUCKET}/*\"
        ]
      },
      {
        \"Sid\": \"AppRunner\",
        \"Effect\": \"Allow\",
        \"Action\": [
          \"apprunner:ListServices\",
          \"apprunner:DescribeService\"
        ],
        \"Resource\": \"*\"
      }
    ]
  }"

# - 5. Create App Runner service
echo ""
echo "→ Creating App Runner service: $APP_RUNNER_SERVICE"

# Push a placeholder image first so the service can be created
echo "  Logging in to ECR..."
aws ecr get-login-password --region "$REGION" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "  Building initial image (this takes a few minutes)..."
docker build -t "${ECR_URI}:latest" .
docker push "${ECR_URI}:latest"

aws apprunner create-service \
  --service-name "$APP_RUNNER_SERVICE" \
  --source-configuration "{
    \"ImageRepository\": {
      \"ImageIdentifier\": \"${ECR_URI}:latest\",
      \"ImageConfiguration\": {
        \"Port\": \"8000\",
        \"RuntimeEnvironmentVariables\": {
          \"MLFLOW_TRACKING_URI\": \"file:///app/mlruns\",
          \"LLM_BACKEND\": \"claude\"
        }
      },
      \"ImageRepositoryType\": \"ECR\"
    },
    \"AutoDeploymentsEnabled\": true,
    \"AuthenticationConfiguration\": {
      \"AccessRoleArn\": \"${ECR_ROLE_ARN}\"
    }
  }" \
  --instance-configuration '{
    "Cpu": "1 vCPU",
    "Memory": "2 GB"
  }' \
  --health-check-configuration '{
    "Protocol": "HTTP",
    "Path": "/health",
    "Interval": 30,
    "Timeout": 10,
    "HealthyThreshold": 1,
    "UnhealthyThreshold": 3
  }' \
  --tags Key=Project,Value=FraudLens Key=Environment,Value=production \
  --region "$REGION" \
  --output json | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('  Service ARN:', d['Service']['ServiceArn'])
print('  Status     :', d['Service']['Status'])
print('  URL        : https://' + d['Service']['ServiceUrl'])
" 2>/dev/null || echo "  (service may already exist — check App Runner console)"

# - 6. Summary 
echo ""
echo "======================================================"
echo " Setup complete. Add these GitHub secrets:"
echo "======================================================"
echo "  AWS_ACCOUNT_ID      = ${ACCOUNT_ID}"
echo "  S3_ARTIFACTS_BUCKET = ${S3_BUCKET}"
echo ""
echo " To redeploy: push to main or run:"
echo "   aws s3 sync mlruns/ s3://${S3_BUCKET}/mlruns/   # after retraining"
echo "   git push origin main                              # triggers CI"
echo "======================================================"
