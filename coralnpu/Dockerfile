FROM node:20-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends chromium git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV PUPPETEER_SKIP_DOWNLOAD=true
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium
ENV NODE_PATH=/antora/node_modules

WORKDIR /antora
COPY package.json .
RUN npm install --omit=dev

COPY entrypoint.sh /antora/entrypoint.sh
RUN chmod +x /antora/entrypoint.sh

WORKDIR /docs
ENTRYPOINT ["/antora/entrypoint.sh"]
CMD ["antora-playbook.yml"]
