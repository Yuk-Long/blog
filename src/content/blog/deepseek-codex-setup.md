---
title: "用 DeepSeek V4 接入 Codex：一次从零开始的配置之旅"
date: 2026-05-29
description: "记录我如何通过 Moon Bridge 将 DeepSeek V4 接入 OpenAI Codex，从环境准备到成功运行的全过程，以及每一步的思考"
tags: ["AI", "Codex", "DeepSeek", "配置"]
readingTime: 8
author: "丝丝大魔王"
---

在搭好这个博客之后，我开始琢磨：能不能用更实惠的模型来驱动我的编程助手？

## 我的想法

Codex 是 OpenAI 出的编程 Agent，非常好用。但默认它连的是 OpenAI 自家的模型，API 的费用对我来说有点高了。在 GitHub 上看到有人用 DeepSeek V4 接入 Codex 的教程，我决定试一试。

简单来说，整个方案就是：

> 我在终端输入指令 → Codex 解析成请求 → Moon Bridge 做翻译转发 → DeepSeek V4 处理 → 结果原路返回给我

之所以需要一个"翻译层"，是因为 Codex 使用 OpenAI Responses API 与模型通信，而 DeepSeek 用的是自己的格式。Moon Bridge 就是那个在中间做格式转换的桥梁。

## 第一步：安装依赖

需要准备三个东西：

- [Node.js](https://nodejs.org/en/download/) 18+
- [Go](https://go.dev/dl/) 1.25+
- Codex CLI

Node.js 我之前搭博客的时候就装好了，所以这一步只需要装 Go 和 Codex：

```shell
npm install -g @openai/codex
```

验证安装：

```shell
codex --version
go version
```

都输出版本号就说明装好了。

## 第二步：获取 DeepSeek API Key

前往 [DeepSeek 开放平台](https://platform.deepseek.com/api_keys) 创建并复制 API Key。

这个 Key 就是你和 DeepSeek 之间的身份凭证，千万不要泄露。

## 第三步：配置 Moon Bridge（最关键的一步）

克隆仓库并进入目录：

```shell
git clone https://github.com/ZhiYi-R/moon-bridge.git
cd moon-bridge
```

创建 `config.yml`，这是我的实际配置：

```yaml
mode: "Transform"

server:
  addr: "127.0.0.1:38441"

models:
  deepseek-v4-pro:
    context_window: 1000000
    max_output_tokens: 384000
    default_reasoning_level: "high"
    supported_reasoning_levels:
      - effort: "high"
        description: "High reasoning effort"
      - effort: "xhigh"
        description: "Extra high reasoning effort"
    supports_reasoning_summaries: true
    default_reasoning_summary: "auto"
    extensions:
      deepseek_v4:
        enabled: true
  deepseek-v4-flash:
    context_window: 1000000
    max_output_tokens: 384000
    default_reasoning_level: "high"
    supported_reasoning_levels:
      - effort: "high"
        description: "High reasoning effort"
      - effort: "xhigh"
        description: "Extra high reasoning effort"
    supports_reasoning_summaries: true
    default_reasoning_summary: "auto"
    extensions:
      deepseek_v4:
        enabled: true

providers:
  deepseek:
    base_url: "https://api.deepseek.com/anthropic"
    api_key: "sk-your-deepseek-api-key"
    offers:
      - model: deepseek-v4-pro
      - model: deepseek-v4-flash

routes:
  moonbridge:
    model: deepseek-v4-pro
    provider: deepseek

defaults:
  model: moonbridge
  max_tokens: 65536
```

我花了不少时间理解每个字段的含义：

- **mode: "Transform"**：运行模式，做请求格式转换
- **server.addr**：我用的端口是 `38441`，因为 38440 被占用了
- **models**：定义你能用哪些模型，以及每个模型的能力边界。上下文窗口 100 万 token，最大输出 384000 token，支持 high 和 xhigh 两种推理档位，还启用了 `deepseek_v4` 扩展
- **providers**：告诉 Moon Bridge 去哪里找模型，base_url 指向 DeepSeek 的 API 地址，api_key 填你的真实 Key
- **routes**：路由规则，把请求转发到 deepseek-v4-pro 模型
- **defaults**：默认模型名为 moonbridge，最大 token 65536

我同时配了两个模型：`deepseek-v4-pro` 处理复杂任务，`deepseek-v4-flash` 用来快速响应简单问题。

## 第四步：启动 Moon Bridge

```shell
go run ./cmd/moonbridge --config config.yml
```

保持这个终端不要关。Moon Bridge 监听 `127.0.0.1:38441`（注意我改了端口），提供 OpenAI Responses 兼容接口：

```text
http://127.0.0.1:38441/v1/responses
```

验证一下是否正常：

```shell
curl http://127.0.0.1:38441/v1/models
```

如果返回了一堆模型信息，说明服务正常。我当时看到返回数据的那一刻还挺激动的——说明链路已经通了一半。

还可以发送一条测试请求：

```shell
curl http://127.0.0.1:38441/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moonbridge",
    "input": "请用一句话打个招呼。",
    "max_output_tokens": 1024
  }'
```

如果能收到正常回复，说明 DeepSeek 那边也通了。

## 第五步：生成 Codex 配置

另开一个终端，在 Moon Bridge 目录下执行。

**如果你已经有 Codex 配置，先备份**：

```powershell
$CODEX_HOME_DIR = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { "$HOME\.codex" }
New-Item -ItemType Directory -Force -Path $CODEX_HOME_DIR | Out-Null

# 备份当前 config.toml
if (Test-Path "$CODEX_HOME_DIR\config.toml") {
  Copy-Item "$CODEX_HOME_DIR\config.toml" "$CODEX_HOME_DIR\config.toml.bak" -Force
}
```

然后生成配置（注意 base-url 用的是 38441 端口）：

```powershell
$MODEL = go run ./cmd/moonbridge --config config.yml --print-codex-model
go run ./cmd/moonbridge `
  --config config.yml `
  --print-codex-config "$MODEL" `
  --codex-base-url "http://127.0.0.1:38441/v1" `
  --codex-home "$CODEX_HOME_DIR" `
  | Set-Content -Path "$CODEX_HOME_DIR\config.toml"
```

这会创建两个关键文件：

- **`config.toml`**：Codex 的 provider 配置，使用 `wire_api = "responses"`，告诉 Codex 去哪找模型
- **`models_catalog.json`**：模型能力清单，包括上下文窗口、推理档位和工具支持等元数据

生成前可以先检查 Moon Bridge 读到的默认模型名称：

```shell
go run ./cmd/moonbridge --config config.yml --print-codex-model
# 输出：moonbridge
```

## 第六步：启动 Codex

进入要处理的项目目录，启动 Codex：

```shell
cd /path/to/my-project
codex
```

此时 Codex 会把 OpenAI Responses 请求发送给本地的 Moon Bridge（端口 38441），再由 Moon Bridge 路由到 DeepSeek V4。整个链路就通了。

包括你现在看到的这篇博客，就是在这种配置下由 Codex + DeepSeek V4 帮我写出来的。

## 我踩过的坑

- **端口冲突**：我一开始用 38440，发现被占用了，于是改成 38441。改完 `config.yml` 后记得同步改第五步里的 `codex-base-url`
- **API Key 不生效（401 错误）**：检查 `config.yml` 中的 API Key 是否粘贴完整，有没有多余空格或引号
- **余额不足（402 错误）**：检查 DeepSeek 开放平台账户余额，充一点就好了
- **Codex 看不到模型**：这个坑我踩了两次——生成完配置后没有重启 Codex。Codex 只在启动时读一次配置，改完必须重启
- **配置加载失败**：提示 `field provider not found`，说明 `config.yml` 格式不对，检查一下是不是用了旧版语法

## 我的感受

整个过程比我想象的要顺利。最让我意外的是 Moon Bridge 的自动配置生成功能——它直接帮我生成了 Codex 需要的所有配置文件，省去了手动查文档、对着模板改的时间。

用 DeepSeek V4 写代码的体验，说实话不比直接用 OpenAI 的模型差。响应速度很快，对中文的理解也很到位。关键是成本低了不止一个数量级，对于个人开发者来说非常友好。

如果你也在用 Codex，强烈建议试试这套配置。按这个顺序来：环境检查 → 获取 Key → 配 Moon Bridge → 生成 Codex 配置 → 启动，每一步都先验证再继续，基本不会出大问题。