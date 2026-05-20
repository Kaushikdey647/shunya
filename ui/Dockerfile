# Multi-stage: build Vite app, serve static assets + /api reverse proxy to FastAPI.
FROM node:22-bookworm AS build

WORKDIR /src
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
ENV VITE_API_BASE=/api
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /src/dist /usr/share/nginx/html
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
