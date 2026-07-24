FROM node:22-alpine
WORKDIR /app
COPY server.mjs ./server.mjs
ENV NODE_ENV=production PORT=8788
EXPOSE 8788
USER node
CMD ["node", "server.mjs"]
