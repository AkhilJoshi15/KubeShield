# KubeShield Data Collection Agent Dockerfile
FROM golang:1.20-alpine AS builder

WORKDIR /app

# Copy source
COPY main.go .

# Build
RUN go build -o kubeshield-collector main.go

# Final image
FROM alpine:3.18

RUN apk --no-cache add ca-certificates

WORKDIR /app

COPY --from=builder /app/kubeshield-collector .

# Run as non-root user
RUN addgroup -g 1000 kubeshield && \
    adduser -D -u 1000 -G kubeshield kubeshield && \
    chown -R kubeshield:kubeshield /app

USER kubeshield

EXPOSE 8080

CMD ["./kubeshield-collector"]
