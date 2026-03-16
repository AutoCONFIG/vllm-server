# =============================================================================
# docker-bake.hcl - vLLM Engine Server Docker构建配置
# =============================================================================
#
# 使用方法:
#   docker buildx bake -f docker/docker-bake.custom.hcl
#
# 参考: https://docs.docker.com/build/bake/reference/
# =============================================================================

variable "VLLM_IMAGE" {
  default = "vllm/vllm-openai:latest"
}

variable "COMMIT" {
  default = ""
}

# =============================================================================
# 目标组
# =============================================================================
group "default" {
  targets = ["vllm-engine-server"]
}

# =============================================================================
# 公共配置
# =============================================================================
target "_common" {
  dockerfile = "docker/Dockerfile.custom"
  context    = "."
  args = {
    VLLM_IMAGE = VLLM_IMAGE
  }
}

target "_labels" {
  labels = {
    "org.opencontainers.image.title"       = "vLLM Engine Server"
    "org.opencontainers.image.description" = "vLLM Engine Server - 高性能LLM推理服务"
    "org.opencontainers.image.version"     = "1.0.0"
    "org.opencontainers.image.licenses"    = "MIT"
    "org.opencontainers.image.revision"    = COMMIT
  }
}

# =============================================================================
# 构建目标
# =============================================================================

# 默认版本 - 基于最新vLLM镜像
target "vllm-engine-server" {
  inherits   = ["_common", "_labels"]
  tags       = ["vllm-engine-server:latest"]
  output     = ["type=docker"]
}

# 指定vLLM版本
target "vllm-engine-server-v0.11.0" {
  inherits   = ["_common", "_labels"]
  args = {
    VLLM_IMAGE = "vllm/vllm-openai:v0.11.0"
  }
  tags       = ["vllm-engine-server:v0.11.0"]
  output     = ["type=docker"]
}
