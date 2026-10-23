+++
date = '2025-04-21T21:46:45+08:00'
draft = false

toc = true

title = 'Using Docker Compose with Dockerfiles'

+++

These notes mainly cover CTF environment configuration, so the examples do not aim for high-performance deployment.

A Dockerfile builds an image; Docker Compose starts and manages multiple containers.

### An example

**Dockerfile**

```bash
# Build stage
FROM python:3.9 as builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production stage
FROM python:3.9-slim

ARG FLAG  # Declare a build argument
ENV FLAG=$FLAG  # Expose it as an environment variable

RUN useradd -m ctfuser
WORKDIR /app
COPY --from=builder /root/.local /app/.local
COPY --chown=ctfuser . .

USER ctfuser

CMD ["python", "app.py"]
```

**docker-compose**

```bash
version: '3.8'  # Version 3.8 is recommended in the original example [3,5](@ref)
services:
  web:
    image: nginx:alpine  # Use a lightweight image
    container_name: my_web  # Custom container name for a single-instance deployment
    ports:
      - "80:80"  # Host port:container port
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro  # Mount the configuration read-only
      - ./static:/var/www/html  # Mount static assets dynamically
    environment:
      - TZ=Asia/Shanghai  # Time zone
    restart: unless-stopped  # Automatic restart policy [7](@ref)
    networks:
      - app_network

networks:
  app_network:
    driver: bridge  # Default network driver
```

### Configuration

(Some Dockerfile settings may be overridden by Docker Compose.)

#### File system

**Dockerfile**

```bash
WORKDIR /app # Created automatically if it does not exist
COPY . /app 
```

**docker-compose**

```bash
volumes:
  - .:/app  
```

#### Environment variables

**Dockerfile**

```bash
ENV KEY=value
```

**docker-compose**

```
    environment:
      - DEBUG=true
```

### Useful commands

```bash
# Build the image without starting containers
docker-compose build

# Start containers using the built image
docker-compose up -d
# Rebuild from scratch without using the cache
docker-compose build --no-cache && docker-compose up -d
```

### Notes

When passing environment variables through docker-compose in this setup, avoid extra spaces and quotation marks.

Do not add quotation marks to the Dockerfile values in this setup.

Use docker exec to run commands inside a container.

