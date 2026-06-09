---
title: "阿里云服务器从零搭建：初始化与环境配置"
date: 2026-06-09
description: "记录我购买第一台阿里云服务器后，从 SSH 登录到环境搭建的完整过程"
tags: ["服务器", "Linux", "阿里云", "运维"]
readingTime: 6
author: "丝丝大魔王"
---

这是我第一次买云服务器，阿里云 ECS，Ubuntu 24.04。记录一下从零开始的每一步。

## 第一步：登录服务器

阿里云控制台的实例列表里找到公网 IP，打开本地终端：

```shell
ssh root@你的公网IP
```

如果你买的时候选了密钥对，找到下载的 `.pem` 文件：

```shell
ssh -i 你的密钥.pem root@你的公网IP
```

**注意**：密钥文件只在购买时能下载，错过了就只能通过控制台重设密码。控制台 → 实例 → 更多 → 密码/密钥 → 重置实例密码 → 重启实例。

连接成功的标志是看到：

```
Welcome to Alibaba Cloud Elastic Compute Service !
```

如果连不上，去阿里云控制台 → 安全组 → 入方向，检查 **22 端口（SSH）** 是不是开放了。没开放就手动加一条：协议 SSH(22)，授权对象 `0.0.0.0/0`。

## 第二步：创建普通用户

不要一直用 root 操作，这是好习惯。先创建一个普通用户：

```shell
adduser dev
```

按提示设密码，其他信息直接回车跳过。然后把它加入 sudo 组：

```shell
usermod -aG sudo dev
```

退出 root，用 dev 重新登录验证：

```shell
exit
ssh dev@你的公网IP
```

测一下 sudo 是否生效：

```shell
sudo whoami
# 应该输出 root
```

> **踩坑**：我第一次 `usermod -aG sudo dev` 是在 dev 用户下执行的，当然没权限。`usermod` 必须 root 才能跑。如果已经退出了 root，可以 `su -` 输入 root 密码切回去。

## 第三步：更新系统

```shell
sudo apt update && sudo apt upgrade -y
```

新系统先更新软件源和所有包，避免后续装软件时出兼容问题。

## 第四步：安装 Nginx

```shell
sudo apt install -y nginx
```

装完之后 Nginx 会自动启动。在浏览器里访问 `http://你的公网IP`，看到 Nginx 欢迎页就说明成功了。

常用命令：

```shell
sudo systemctl status nginx   # 查看状态
sudo systemctl start nginx    # 启动
sudo systemctl stop nginx     # 停止
sudo systemctl restart nginx  # 重启
```

## 第五步：安装 Node.js（nvm）

用 nvm 管理 Node 版本，比直接 apt 装灵活得多：

```shell
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
```

装完后退出重新登录，然后安装 Node：

```shell
nvm install 22
node -v   # 验证
npm -v    # 验证
```

---

到这里服务器的基础环境就搭好了。Nginx 跑起来、Node.js 装上，下一步就是把博客代码传上去并部署。

*（未完待续）*
