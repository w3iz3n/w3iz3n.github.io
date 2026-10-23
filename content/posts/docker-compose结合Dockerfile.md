+++
date = '2025-04-21T21:46:45+08:00'
draft = false

toc = true

title = 'Docker Compose结合Dockerfile'

+++

主要是在做CTF环境配置中的记录 所以没有多高性能要求的配置项

Dockerfile用来构建镜像 docker-compose用来启动管理多容器

### 一个案例

**Dockerfile**

```bash
# 构建阶段
FROM python:3.9 as builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# 生产阶段
FROM python:3.9-slim

ARG FLAG  # 构建参数声明
ENV FLAG=$FLAG  # 转为环境变量

RUN useradd -m ctfuser
WORKDIR /app
COPY --from=builder /root/.local /app/.local
COPY --chown=ctfuser . .

USER ctfuser

CMD ["python", "app.py"]
```

**docker-compose**

```bash
version: '3.8'  # 推荐使用 3.8 版本[3,5](@ref)
services:
  web:
    image: nginx:alpine  # 使用轻量级镜像
    container_name: my_web  # 自定义容器名（单实例场景）
    ports:
      - "80:80"  # 主机端口:容器端口
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro  # 只读挂载配置
      - ./static:/var/www/html  # 动态挂载静态资源
    environment:
      - TZ=Asia/Shanghai  # 时区设置
    restart: unless-stopped  # 自动重启策略[7](@ref)
    networks:
      - app_network

networks:
  app_network:
    driver: bridge  # 默认网络驱动
```

### 配置

（有些Dockerfile中的配置会被docker-compose中的覆盖）

#### 文件系统

**Dockerfile**

```bash
WORKDIR /app #(没有的话会自动创建)
COPY . /app 
```

**docker-compose**

```bash
volumes:
  - .:/app  
```

#### 环境的变量

**Dockerfile**

```bash
ENV KEY=value
```

**docker-compose**

```
    environment:
      - DEBUG=true
```

### 一些用到的命令

```bash
# 构建镜像（不启动容器）
docker-compose build

# 启动容器（使用已构建的镜像）
docker-compose up -d
# 完全重新构建（清除缓存）
docker-compose build --no-cache && docker-compose up -d
```

### 注意事项

docker-compose传递环境变量不要加空格和引号

Dockerfile不要加引号

docker exec 可以执行容器里的命令

