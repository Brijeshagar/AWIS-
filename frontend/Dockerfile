# =============================================================================
# AWIS Frontend — React + Vite → Nginx
#
# NOTE: Build context is the REPOSITORY ROOT (set in docker-compose.yml).
#       All paths below are relative to that root.
# =============================================================================

# Stage 1: builder — compile the React app with Vite
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package manifests first for layer caching:
# npm install only re-runs if package.json or package-lock.json change.
COPY frontend/package.json frontend/package-lock.json ./

RUN npm ci --prefer-offline

# Copy the rest of the frontend source
COPY frontend/ .

# VITE_API_URL is baked into the compiled JS bundle at build time.
# In docker-compose the backend service is reachable as http://backend:8000
# WITHIN the container-to-container network, but the BROWSER calls the backend
# from the host machine — so we expose the backend on host port 8000 and use
# http://localhost:8000 as the default here (override for prod deployments).
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# =============================================================================
# Stage 2: runtime — serve compiled assets with Nginx (tiny alpine image ~25MB)
# =============================================================================
FROM nginx:1.27-alpine AS runtime

# Remove Nginx default welcome page
RUN rm -rf /usr/share/nginx/html/*

# Copy compiled React dist output from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy our custom Nginx server config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
