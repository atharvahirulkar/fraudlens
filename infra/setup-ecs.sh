#!/usr/bin/env bash
# One-time AWS infrastructure bootstrap for FraudLens — ECS Fargate edition.
# Drop-in replacement for setup.sh; use this instead of App Runner.
#
# Prerequisites:
#   aws configure   (or set AWS_PROFILE)
#   docker running
#
# Usage:
#   chmod +x infra/setup-ecs.sh
#   GITHUB_ORG=your-github-username GITHUB_REPO=fraudlens bash infra/setup-ecs.sh

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="fraudlens"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"
S3_BUCKET="fraudlens-artifacts-${ACCOUNT_ID}"
CLUSTER="fraudlens"
SERVICE="fraudlens-inference"
TASK_FAMILY="fraudlens-inference"
GITHUB_ORG="${GITHUB_ORG:?Set GITHUB_ORG to your GitHub username/org}"
GITHUB_REPO="${GITHUB_REPO:-fraudlens}"

echo "======================================================"
echo " FraudLens AWS Bootstrap — ECS Fargate"
echo " Account : $ACCOUNT_ID"
echo " Region  : $REGION"
echo " S3      : $S3_BUCKET"
echo "======================================================"

# ── 1. ECR repository (skip if exists) ─────────────────────────────────────
echo ""
echo "→ ECR repository: $ECR_REPO"
aws ecr create-repository \
  --repository-name "$ECR_REPO" \
  --region "$REGION" \
  --image-scanning-configuration scanOnPush=true \
  --output json 2>/dev/null \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('  Created:', d['repository']['repositoryUri'])" \
  || echo "  (already exists)"

# ── 2. S3 bucket (skip if exists) ──────────────────────────────────────────
echo ""
echo "→ S3 bucket: $S3_BUCKET"
aws s3api create-bucket --bucket "$S3_BUCKET" --region "$REGION" 2>/dev/null || echo "  (already exists)"
aws s3api put-bucket-versioning \
  --bucket "$S3_BUCKET" \
  --versioning-configuration Status=Enabled

echo ""
echo "→ Uploading model artifacts to S3"
aws s3 sync mlruns/ "s3://${S3_BUCKET}/mlruns/" --region "$REGION"
echo "  s3://${S3_BUCKET}/mlruns/"

# ── 3. Security groups ──────────────────────────────────────────────────────
VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" \
  --filters Name=isDefault,Values=true \
  --query "Vpcs[0].VpcId" --output text)

echo ""
echo "→ VPC: $VPC_ID"

# ALB security group
ALB_SG=$(aws ec2 describe-security-groups --region "$REGION" \
  --filters "Name=group-name,Values=fraudlens-alb" "Name=vpc-id,Values=${VPC_ID}" \
  --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || true)

if [ "$ALB_SG" = "None" ] || [ -z "$ALB_SG" ]; then
  echo "→ Creating ALB security group"
  ALB_SG=$(aws ec2 create-security-group \
    --group-name fraudlens-alb \
    --description "FraudLens ALB" \
    --vpc-id "$VPC_ID" \
    --region "$REGION" \
    --query GroupId --output text)
  aws ec2 authorize-security-group-ingress \
    --group-id "$ALB_SG" --region "$REGION" \
    --ip-permissions \
      "IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges=[{CidrIp=0.0.0.0/0}]" \
      "IpProtocol=tcp,FromPort=443,ToPort=443,IpRanges=[{CidrIp=0.0.0.0/0}]"
else
  echo "→ ALB security group exists: $ALB_SG"
fi

# ECS task security group
ECS_SG=$(aws ec2 describe-security-groups --region "$REGION" \
  --filters "Name=group-name,Values=fraudlens-ecs" "Name=vpc-id,Values=${VPC_ID}" \
  --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || true)

if [ "$ECS_SG" = "None" ] || [ -z "$ECS_SG" ]; then
  echo "→ Creating ECS task security group"
  ECS_SG=$(aws ec2 create-security-group \
    --group-name fraudlens-ecs \
    --description "FraudLens ECS tasks" \
    --vpc-id "$VPC_ID" \
    --region "$REGION" \
    --query GroupId --output text)
  aws ec2 authorize-security-group-ingress \
    --group-id "$ECS_SG" --region "$REGION" \
    --ip-permissions \
      "IpProtocol=tcp,FromPort=8000,ToPort=8000,UserIdGroupPairs=[{GroupId=$ALB_SG}]"
else
  echo "→ ECS security group exists: $ECS_SG"
fi

echo "  ALB SG : $ALB_SG"
echo "  ECS SG : $ECS_SG"

# ── 4. Application Load Balancer ────────────────────────────────────────────
SUBNETS=$(aws ec2 describe-subnets --region "$REGION" \
  --filters "Name=vpc-id,Values=${VPC_ID}" Name=defaultForAz,Values=true \
  --query "Subnets[*].SubnetId" --output text | tr '\t' ',')

ALB_ARN=$(aws elbv2 describe-load-balancers --region "$REGION" \
  --names fraudlens-alb \
  --query "LoadBalancers[0].LoadBalancerArn" --output text 2>/dev/null || true)

if [ "$ALB_ARN" = "None" ] || [ -z "$ALB_ARN" ]; then
  echo ""
  echo "→ Creating Application Load Balancer"
  ALB_ARN=$(aws elbv2 create-load-balancer \
    --name fraudlens-alb \
    --subnets $(echo "$SUBNETS" | tr ',' ' ') \
    --security-groups "$ALB_SG" \
    --region "$REGION" \
    --query "LoadBalancers[0].LoadBalancerArn" --output text)
  echo "  ALB ARN: $ALB_ARN"
else
  echo ""
  echo "→ ALB exists: $ALB_ARN"
fi

ALB_DNS=$(aws elbv2 describe-load-balancers --region "$REGION" \
  --load-balancer-arns "$ALB_ARN" \
  --query "LoadBalancers[0].DNSName" --output text)
echo "  ALB DNS: $ALB_DNS"

# ── 5. Target group ─────────────────────────────────────────────────────────
TG_ARN=$(aws elbv2 describe-target-groups --region "$REGION" \
  --names fraudlens-api \
  --query "TargetGroups[0].TargetGroupArn" --output text 2>/dev/null || true)

if [ "$TG_ARN" = "None" ] || [ -z "$TG_ARN" ]; then
  echo ""
  echo "→ Creating target group"
  TG_ARN=$(aws elbv2 create-target-group \
    --name fraudlens-api \
    --protocol HTTP \
    --port 8000 \
    --vpc-id "$VPC_ID" \
    --target-type ip \
    --health-check-path /health \
    --health-check-interval-seconds 30 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 3 \
    --region "$REGION" \
    --query "TargetGroups[0].TargetGroupArn" --output text)
else
  echo ""
  echo "→ Target group exists"
fi
echo "  TG ARN: $TG_ARN"

# ── 6. ALB listener ─────────────────────────────────────────────────────────
LISTENER_COUNT=$(aws elbv2 describe-listeners --region "$REGION" \
  --load-balancer-arn "$ALB_ARN" \
  --query "length(Listeners)" --output text 2>/dev/null || echo 0)

if [ "$LISTENER_COUNT" = "0" ]; then
  echo ""
  echo "→ Creating ALB listener (port 80)"
  aws elbv2 create-listener \
    --load-balancer-arn "$ALB_ARN" \
    --protocol HTTP \
    --port 80 \
    --default-actions "Type=forward,TargetGroupArn=${TG_ARN}" \
    --region "$REGION" > /dev/null
else
  echo "→ ALB listener exists"
fi

# ── 7. ECS cluster ──────────────────────────────────────────────────────────
echo ""
echo "→ Creating ECS cluster: $CLUSTER"
aws ecs create-cluster \
  --cluster-name "$CLUSTER" \
  --capacity-providers FARGATE FARGATE_SPOT \
  --region "$REGION" \
  --output json 2>/dev/null \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('  Status:', d['cluster']['status'])" \
  || echo "  (already exists)"

# ── 8. ECS task execution role ──────────────────────────────────────────────
echo ""
echo "→ Creating ECS task execution role: FraudLensECSExecutionRole"
aws iam create-role \
  --role-name FraudLensECSExecutionRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ecs-tasks.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }' 2>/dev/null || echo "  (already exists)"

aws iam attach-role-policy \
  --role-name FraudLensECSExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy \
  2>/dev/null || true

# Allow reading the OPENAI_API_KEY SSM parameter
aws iam put-role-policy \
  --role-name FraudLensECSExecutionRole \
  --policy-name FraudLensSSMRead \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [{
      \"Effect\": \"Allow\",
      \"Action\": [\"ssm:GetParameters\", \"kms:Decrypt\"],
      \"Resource\": \"arn:aws:ssm:${REGION}:${ACCOUNT_ID}:parameter/fraudlens/*\"
    }]
  }"

ECS_EXEC_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/FraudLensECSExecutionRole"

# ── 9. SSM parameter for OPENAI_API_KEY ─────────────────────────────────────
echo ""
echo "→ Checking SSM parameter /fraudlens/OPENAI_API_KEY"
PARAM_EXISTS=$(aws ssm get-parameter \
  --name "/fraudlens/OPENAI_API_KEY" \
  --region "$REGION" \
  --query "Parameter.Name" --output text 2>/dev/null || true)

if [ -z "$PARAM_EXISTS" ]; then
  echo "  Creating placeholder — UPDATE THIS with your real key:"
  echo "  aws ssm put-parameter --name /fraudlens/OPENAI_API_KEY --value 'sk-...' --type SecureString --overwrite --region $REGION"
  aws ssm put-parameter \
    --name "/fraudlens/OPENAI_API_KEY" \
    --value "REPLACE_ME" \
    --type SecureString \
    --region "$REGION" > /dev/null
else
  echo "  Already set."
fi

# ── 10. CloudWatch log group ────────────────────────────────────────────────
echo ""
echo "→ Creating CloudWatch log group: /ecs/${SERVICE}"
aws logs create-log-group \
  --log-group-name "/ecs/${SERVICE}" \
  --region "$REGION" 2>/dev/null || echo "  (already exists)"

# ── 11. Task definition ─────────────────────────────────────────────────────
echo ""
echo "→ Registering ECS task definition: $TASK_FAMILY"

# Build initial image and push so the task def points to a real image
echo "  Logging in to ECR..."
aws ecr get-login-password --region "$REGION" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "  Building Docker image..."
docker build -t "${ECR_URI}:latest" .
docker push "${ECR_URI}:latest"
echo "  Pushed: ${ECR_URI}:latest"

TASK_DEF_ARN=$(aws ecs register-task-definition \
  --family "$TASK_FAMILY" \
  --network-mode awsvpc \
  --requires-compatibilities FARGATE \
  --cpu 1024 \
  --memory 2048 \
  --execution-role-arn "$ECS_EXEC_ROLE_ARN" \
  --region "$REGION" \
  --container-definitions "[
    {
      \"name\": \"fraudlens-api\",
      \"image\": \"${ECR_URI}:latest\",
      \"portMappings\": [{\"containerPort\": 8000, \"protocol\": \"tcp\"}],
      \"environment\": [
        {\"name\": \"MLFLOW_TRACKING_URI\", \"value\": \"file:///app/mlruns\"},
        {\"name\": \"LLM_BACKEND\", \"value\": \"openai\"}
      ],
      \"secrets\": [
        {\"name\": \"OPENAI_API_KEY\", \"valueFrom\": \"arn:aws:ssm:${REGION}:${ACCOUNT_ID}:parameter/fraudlens/OPENAI_API_KEY\"}
      ],
      \"logConfiguration\": {
        \"logDriver\": \"awslogs\",
        \"options\": {
          \"awslogs-group\": \"/ecs/${SERVICE}\",
          \"awslogs-region\": \"${REGION}\",
          \"awslogs-stream-prefix\": \"ecs\"
        }
      },
      \"healthCheck\": {
        \"command\": [\"CMD-SHELL\", \"curl -f http://localhost:8000/health || exit 1\"],
        \"interval\": 30,
        \"timeout\": 5,
        \"retries\": 3,
        \"startPeriod\": 60
      }
    }
  ]" \
  --query "taskDefinition.taskDefinitionArn" --output text)

echo "  Task def: $TASK_DEF_ARN"

# ── 12. ECS service ─────────────────────────────────────────────────────────
echo ""
echo "→ Creating ECS service: $SERVICE"

SUBNET_LIST=$(aws ec2 describe-subnets --region "$REGION" \
  --filters "Name=vpc-id,Values=${VPC_ID}" Name=defaultForAz,Values=true \
  --query "Subnets[*].SubnetId" --output json)

aws ecs create-service \
  --cluster "$CLUSTER" \
  --service-name "$SERVICE" \
  --task-definition "$TASK_FAMILY" \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "{
    \"awsvpcConfiguration\": {
      \"subnets\": ${SUBNET_LIST},
      \"securityGroups\": [\"${ECS_SG}\"],
      \"assignPublicIp\": \"ENABLED\"
    }
  }" \
  --load-balancers "[
    {
      \"targetGroupArn\": \"${TG_ARN}\",
      \"containerName\": \"fraudlens-api\",
      \"containerPort\": 8000
    }
  ]" \
  --health-check-grace-period-seconds 120 \
  --region "$REGION" \
  --output json 2>/dev/null \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('  Service:', d['service']['serviceArn'])" \
  || echo "  (service already exists)"

# ── 13. GitHub OIDC provider + GitHubActionsRole ────────────────────────────
echo ""
echo "→ OIDC provider for GitHub Actions"
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 \
  2>/dev/null || echo "  (already exists)"

OIDC_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"

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
        \"Sid\": \"ECSdeploy\",
        \"Effect\": \"Allow\",
        \"Action\": [
          \"ecs:RegisterTaskDefinition\",
          \"ecs:DescribeTaskDefinition\",
          \"ecs:DeregisterTaskDefinition\",
          \"ecs:UpdateService\",
          \"ecs:DescribeServices\",
          \"ecs:DescribeClusters\",
          \"ecs:ListTasks\",
          \"ecs:DescribeTasks\"
        ],
        \"Resource\": \"*\"
      },
      {
        \"Sid\": \"IAMPassRole\",
        \"Effect\": \"Allow\",
        \"Action\": \"iam:PassRole\",
        \"Resource\": \"arn:aws:iam::${ACCOUNT_ID}:role/FraudLensECSExecutionRole\"
      },
      {
        \"Sid\": \"ELB\",
        \"Effect\": \"Allow\",
        \"Action\": [
          \"elasticloadbalancing:DescribeTargetHealth\",
          \"elasticloadbalancing:DescribeTargetGroups\"
        ],
        \"Resource\": \"*\"
      }
    ]
  }"

echo "  GitHubActionsRole updated with ECS permissions"

# ── 14. Summary ─────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " Setup complete!"
echo "======================================================"
echo ""
echo " IMPORTANT — set your OpenAI key:"
echo "   aws ssm put-parameter \\"
echo "     --name /fraudlens/OPENAI_API_KEY \\"
echo "     --value 'sk-...' \\"
echo "     --type SecureString --overwrite \\"
echo "     --region $REGION"
echo ""
echo " Add these GitHub secrets:"
echo "   AWS_ACCOUNT_ID      = ${ACCOUNT_ID}"
echo "   S3_ARTIFACTS_BUCKET = ${S3_BUCKET}"
echo ""
echo " Your API will be reachable at:"
echo "   http://${ALB_DNS}/health   (HTTP; ~3 min for first task to start)"
echo ""
echo " To redeploy: push to main branch"
echo "   git push origin main"
echo "======================================================"
