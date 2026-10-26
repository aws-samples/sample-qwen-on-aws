# SageMaker HyperPod (EKS) — on-policy distillation reference stack

A single, self-contained CloudFormation template that stands up a SageMaker HyperPod
cluster orchestrated by Amazon EKS, with the storage, observability and notebook
tooling an on-policy distillation workload needs.

Template: [`hyperpod-eks-onpolicy-distillation.yaml`](./hyperpod-eks-onpolicy-distillation.yaml)

## What it creates

| Area | Resources |
| --- | --- |
| Network | VPC (`10.192.0.0/16`) + 4 secondary CIDRs, 4 public /24s, 4 EKS control-plane /24s, one full **/16 HyperPod subnet per AZ**, IGW, single NAT gateway, S3 gateway endpoint, interface endpoints for AMP and Grafana, one shared security group |
| Storage | S3 data bucket + S3 access-log bucket (AES256, public access blocked, TLS-only bucket policies); FSx for Lustre `PERSISTENT_2` 1200 GiB @ 250 MB/s/TiB, LZ4, Lustre 2.15 |
| Orchestration | EKS cluster (Kubernetes 1.34 by default — the minimum for the task-governance add-on's Kueue; empty `KubernetesVersion` = latest standard-support version; `API_AND_CONFIG_MAP`, all 5 control-plane log types, public + private endpoint) |
| EKS add-ons | `vpc-cni`, `kube-proxy`, `coredns`, `eks-pod-identity-agent`, `metrics-server`, `cert-manager`, `aws-ebs-csi-driver`, `aws-fsx-csi-driver`, `aws-mountpoint-s3-csi-driver`, `amazon-cloudwatch-observability`, `amazon-sagemaker-hyperpod-observability`, `amazon-sagemaker-hyperpod-taskgovernance`, `amazon-sagemaker-spaces` |
| Cluster deps | HyperPod Helm chart from [`aws/sagemaker-hyperpod-cli`](https://github.com/aws/sagemaker-hyperpod-cli) (device plugins, EFA plugin, health-monitoring agent, deep health check, job auto-restart, training + MPI operators, MLflow) installed by a CodeBuild-backed custom resource |
| HyperPod | `AWS::SageMaker::Cluster` named **`hp-cluster-onpolicy-distillation`**, two instance groups, `NodeRecovery: Automatic`, `NodeProvisioningMode: Continuous`, Karpenter autoscaling |
| Kubernetes storage | Static `PersistentVolume`/`PersistentVolumeClaim` pairs: `fsx-pv`/`fsx-claim` and `s3-pv`/`s3-claim` in `default` |
| Observability | Amazon Managed Prometheus workspace, optional Amazon Managed Grafana workspace (Prometheus + CloudWatch data sources), scoped IAM roles |
| Notebooks | New SageMaker Studio domain + user profile, execution role with EKS/HyperPod/S3/FSx/AMP access, EKS access entry, spaces controller and SSM managed-node roles for remote access into HyperPod-hosted spaces |
| Notebook bootstrap | JupyterLab **Studio lifecycle configuration** (`<prefix>-clone-repo`) that clones `GitRepositoryUrl` into the space and writes `~/opd-env.sh`; a private JupyterLab **space** (`opd-jupyterlab`) owned by the user profile, with the repository registered as a code repository |

### Instance groups

| Group | Default type | Count | Notes |
| --- | --- | --- | --- |
| `general-purpose` | `ml.m5.4xlarge` | 2 | Spread across all four HyperPod subnets |
| `accelerated` | `ml.g5.24xlarge` | 1 | 500 GiB EBS, pinned via `OverrideVpcConfig` to the same AZ as FSx for Lustre |

## Prerequisites

1. **HyperPod capacity.** The accelerated instance type must be available to your
   account in the target region — request a HyperPod cluster quota increase for
   `ml.g5.24xlarge` (or whichever type you pick) before deploying. Without quota,
   the `AWS::SageMaker::Cluster` resource fails and rolls the stack back.
2. **AWS IAM Identity Center** must be enabled in the deployment region if you leave
   `CreateGrafanaWorkspace=true`. Amazon Managed Grafana's `AWS_SSO` authentication
   provider has no fallback; if Identity Center is not enabled, stack creation fails.
   Set `CreateGrafanaWorkspace=false` to skip Grafana — Prometheus and the
   observability add-on still work, and you can point your own Grafana at the AMP
   endpoint from the `PrometheusEndpoint` output.
3. **Four Availability Zones.** The template uses `Fn::GetAZs` and assumes at least
   four AZs in the region.
4. **Outbound internet** from the HyperPod subnets (provided by the NAT gateway) so
   nodes can pull container images and the CodeBuild bootstrap can clone the Helm chart.
5. `CAPABILITY_NAMED_IAM` — the template creates named IAM roles and a managed policy.

## Deploy

The template is larger than the 51,200-byte inline limit, so pass `--s3-bucket <bucket>`
(any bucket you can write to) and let the CLI stage it.

```bash
aws cloudformation deploy \
  --stack-name hp-onpolicy-distillation \
  --template-file infra/hyperpod-eks-onpolicy-distillation.yaml \
  --s3-bucket <staging-bucket> \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-west-2 \
  --parameter-overrides \
      ResourceNamePrefix=hp-onpolicy-distill \
      HyperPodClusterName=hp-cluster-onpolicy-distillation \
      AcceleratedInstanceType=ml.g5.24xlarge \
      AcceleratedInstanceCount=1 \
      GitRepositoryUrl=<your-git-repo-url> \
      AdminPrincipalArn=arn:aws:iam::<account-id>:role/<your-role>
```

Without Identity Center, add `CreateGrafanaWorkspace=false`.

Expect roughly 45–70 minutes end to end: the EKS cluster and add-ons take ~15 min,
FSx ~10 min, the Helm bootstrap ~5 min, and the HyperPod cluster ~15–25 min
depending on how quickly capacity is allocated.

## After deployment

```bash
# Point kubectl at the cluster (also emitted as the UpdateKubeconfigCommand output)
aws eks update-kubeconfig --region us-west-2 --name <EKSClusterName>

kubectl get nodes -L sagemaker.amazonaws.com/instance-group-name
kubectl get pvc            # fsx-claim and s3-claim should be Bound
aws sagemaker describe-cluster --cluster-name hp-cluster-onpolicy-distillation
```

Mount the shared storage in a training pod:

```yaml
volumes:
  - name: shared-fsx
    persistentVolumeClaim: {claimName: fsx-claim}
  - name: shared-s3
    persistentVolumeClaim: {claimName: s3-claim}
```

For Studio, open the `StudioDomainUrl` output and launch the `opd-jupyterlab` space as
the `distillation-researcher` user profile. The lifecycle configuration clones
`GitRepositoryUrl` into `~/<repo-name>` and writes `~/opd-env.sh` on every app start —
see the root [`README.md`](../README.md) for the full runbook. Because the domain
execution role holds an EKS access entry, `kubectl` and the `hyperpod` CLI work from a
notebook terminal after running `update-kubeconfig`.

`CreateSpace` cannot target HyperPod compute, so the CloudFormation-created space runs on a
Studio-managed instance (`StudioSpaceInstanceType`). Spaces that run *on* the cluster are a
different resource: the `amazon-sagemaker-spaces` add-on registers a `Workspace` CRD
(`workspace.jupyter.org/v1alpha1`) and you create them with the HyperPod CLI, `kubectl`, or
the HyperPod console — not with the SageMaker API, and with no Studio domain involved:

```bash
hyp create hyp-space --name opd-space --display-name opd-space \
  --template-ref name=sagemaker-jupyter-template,namespace=jupyter-k8s-system \
  --node-selector '{"node.kubernetes.io/instance-type":"ml.m5.4xlarge"}' \
  --cpu 4 --memory 16Gi \
  --volume name=fsx-data,mountPath=/fsx,persistentVolumeClaimName=fsx-claim \
  --volume name=s3-data,mountPath=/s3,persistentVolumeClaimName=s3-claim
hyp portforward hyp-space --name opd-space --local-port 8888
```

The Studio lifecycle configuration does not apply to those; the root README documents the
`postStart` alternative. Spaces created this way are not part of the stack — delete them
with `hyp delete hyp-space` before deleting the cluster.

## Outputs

Required by the sample consumers:

| Output | Description |
| --- | --- |
| `HyperPodClusterName` | HyperPod cluster name |
| `HyperPodClusterArn` | HyperPod cluster ARN |
| `EKSClusterName` | Orchestrating EKS cluster name |
| `EKSClusterArn` | Orchestrating EKS cluster ARN |
| `S3BucketName` / `S3BucketArn` | Shared S3 bucket |

Also exported: `EKSClusterEndpoint`, `EKSOidcIssuerUrl`, `UpdateKubeconfigCommand`,
`S3AccessLogsBucketName`, `S3BucketAccessPolicyArn`, `FsxFileSystemId`,
`FsxMountName`, `FsxDnsName`, `PrometheusWorkspaceId`, `PrometheusWorkspaceArn`,
`PrometheusEndpoint`, `GrafanaWorkspaceEndpoint`, `GrafanaWorkspaceId`,
`StudioDomainId`, `StudioDomainUrl`, `StudioExecutionRoleArn`,
`StudioUserProfileName`, `StudioLifecycleConfigArn`, `NotebookRepositoryUrl`,
`StudioSpaceName`, `StudioSpaceArn`, `StudioSpaceJupyterLabUrl`, `VpcId`,
`SecurityGroupId`, `HyperPodSubnetIds`, `EksSubnetIds`,
`HyperPodExecutionRoleArn`, `HyperPodClusterRoleArn`.

The five required outputs plus the bucket ARN are also CloudFormation exports named
`<stack-name>-<OutputName>`, so downstream stacks can `Fn::ImportValue` them.

## S3 bucket permissions

The bucket is named `<ResourceNamePrefix>-<AccountId>-bucket`. Access is granted in
two directions so both the EKS/Mountpoint path and the Studio path work:

**IAM side** — one managed policy, `<prefix>-bucket-access` (output
`S3BucketAccessPolicyArn`), holding exactly what Mountpoint for Amazon S3 requires:

- `s3:ListBucket`, `s3:GetBucketLocation` on the bucket
- `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`, `s3:AbortMultipartUpload`,
  `s3:ListMultipartUploadParts` on `bucket/*`

It is attached to three roles:

| Role | Why |
| --- | --- |
| `<prefix>-s3-csi-role` | Mountpoint S3 CSI driver, wired by **EKS Pod Identity** to `kube-system/s3-csi-driver-sa` — this is what backs `s3-pv`/`s3-claim` |
| `<prefix>-exec-role` | HyperPod node instance role — reads `on_create.sh`, writes checkpoints |
| `<prefix>-studio-exec-role` | SageMaker Studio spaces |

**Bucket side** — the bucket policy denies non-TLS access and explicitly allows those
same three principals, so the grant is legible from the bucket as well as from IAM.

Unlike the console-generated reference configuration, which grants Mountpoint
`s3:*` object actions on `arn:aws:s3:::*`, this policy is scoped to the single
bucket the stack creates.

## Design notes

- **EKS Pod Identity, not IRSA.** All in-cluster roles (`aws-ebs-csi-driver`,
  `aws-fsx-csi-driver`, `aws-mountpoint-s3-csi-driver`, the HyperPod observability
  collector, the spaces controller) use `pods.eks.amazonaws.com` trust with
  `PodIdentityAssociations` on the add-on. That removes the need for an
  `AWS::IAM::OIDCProvider` and its certificate thumbprint, which is not something a
  plain CloudFormation template can compute.
- **FSx is statically provisioned.** The file system is created by CloudFormation and
  surfaced to Kubernetes as a pre-bound `PersistentVolume`, so the CSI controller
  never needs `fsx:CreateFileSystem` at runtime for the sample path.
- **AZ pinning.** `PrimaryAvailabilityZoneIndex` selects the AZ that hosts both FSx
  and the accelerated instance group, keeping Lustre traffic in-AZ. Change it if
  your GPU capacity lives elsewhere.
- **Addressing.** HyperPod subnets each get a whole secondary /16
  (`10.1.0.0/16`–`10.4.0.0/16`) because pod ENIs and EFA interfaces on large
  training fleets exhaust /24s quickly. These four ranges are hardcoded — if you
  change `VpcCIDR` to something inside `10.1.0.0/14`, edit the `HyperPodCidr*`
  resources to avoid an overlap.
- **Add-on versions are not pinned.** Each `AWS::EKS::Addon` omits `AddonVersion` so
  EKS installs the default for the cluster's Kubernetes version. Pin versions if you
  need byte-for-byte reproducibility.
- **Repository cloning is a lifecycle configuration, not a code repository.** A
  `CodeRepositories` entry only *offers* a repository for cloning in the JupyterLab git
  panel; it does not clone anything. The stack therefore does both: it registers the
  repository (domain, user profile and space) and attaches a JupyterLab
  `AWS::SageMaker::StudioLifecycleConfig` that actually clones it. The script is
  idempotent (fast-forwards an existing checkout), fully guarded with `|| true`, and ends
  in `exit 0` so a network hiccup cannot stop the space from starting; it logs to
  `~/.opd-lifecycle.log`.
- **Spaces add-on configuration.** `SageMakerSpacesAddon` enables
  `jupyter-k8s.workspacePodWatching` and `jupyter-k8s-aws-hyperpod.remoteAccess` (pointing at
  `SpaceSsmManagedNodeRole`), so remote IDE connections into HyperPod spaces work.
  `clusterWebUI` is left at its default of `false` — turning it on additionally requires a DNS
  domain, an ACM certificate and the bundled Traefik ingress. Until then, reach a HyperPod
  space with `hyp portforward` or a remote IDE. Run
  `aws eks describe-addon-configuration --addon-name amazon-sagemaker-spaces --addon-version <v>`
  to see the full schema.
- **Helm bootstrap ordering.** `HelmDependencies` must finish before the HyperPod
  cluster is created — nodes will not reach `InService` without the health-monitoring
  agent and device plugins present. `StorageBootstrap` runs afterwards, once both
  FSx and the cluster exist. Both are CodeBuild builds driven by one Lambda-backed
  custom resource; failures show up in the `/aws/codebuild/<prefix>-bootstrap` log
  group.

## Not included

The HyperPod **inference operator** stack is deliberately omitted: the AWS Load
Balancer Controller, KEDA, the TLS-certificate bucket, JumpStart gated-model access
and restricted instance groups (RIG). This sample targets training and distillation.
Add them if you plan to serve endpoints from the same cluster.

## Cleanup

Stop the JupyterLab app in the Studio UI first — a running app blocks deletion of the
space, which blocks deletion of the domain and therefore the stack.

```bash
aws cloudformation delete-stack --stack-name hp-onpolicy-distillation --region us-west-2
```

Retained on purpose (delete manually if you want them gone):

- the S3 data bucket and access-log bucket (`DeletionPolicy: Retain`, and a
  non-empty bucket cannot be deleted by CloudFormation anyway)
- the Amazon Managed Prometheus workspace (`DeletionPolicy: Retain`)

The FSx for Lustre file system is **deleted** with the stack, and it has no
automatic backup configured — copy anything you need off `fsx-claim` first.

## Cost

This is not a small stack. At `us-west-2` on-demand pricing the standing cost is
dominated by the HyperPod instance groups (2× `ml.m5.4xlarge` + 1× `ml.g5.24xlarge`),
then FSx for Lustre (1200 GiB `PERSISTENT_2` @ 250 MB/s/TiB), the EKS control plane,
the NAT gateway and its data processing, Amazon Managed Grafana per-user licensing,
and AMP metric ingestion and storage. Scale
`GeneralPurposeInstanceCount`/`AcceleratedInstanceCount` to 1 and set
`CreateGrafanaWorkspace=false` while you are just validating the deployment.
