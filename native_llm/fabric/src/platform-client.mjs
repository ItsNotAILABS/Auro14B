const DEFAULT_TIMEOUT_MS = 30_000;

function normalizeBaseUrl(value) {
  if (!value) return null;
  return value.endsWith('/') ? value.slice(0, -1) : value;
}

export function createPlatformClient({
  baseUrl = normalizeBaseUrl(process.env.CLOUDFLARE_PLATFORM_URL),
  token = process.env.CLOUDFLARE_PLATFORM_TOKEN,
  timeoutMs = DEFAULT_TIMEOUT_MS,
} = {}) {
  return {
    describe() {
      return {
        mode: baseUrl ? 'remote-cloudflare' : 'unconfigured',
        base_url_configured: Boolean(baseUrl),
        token_configured: Boolean(token),
      };
    },

    async request(path, options = {}) {
      if (!baseUrl) {
        return {
          status: 503,
          ok: false,
          error: 'cloudflare_platform_not_configured',
        };
      }

      if (typeof path !== 'string' || !path.startsWith('/')) {
        return {
          status: 400,
          ok: false,
          error: 'invalid_platform_path',
        };
      }

      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const headers = new Headers(options.headers ?? {});
        headers.set('accept', 'application/json');
        if (token) headers.set('authorization', `Bearer ${token}`);
        let body;
        if (options.body !== undefined) {
          headers.set('content-type', 'application/json');
          body = JSON.stringify(options.body);
        }

        const response = await fetch(`${baseUrl}${path}`, {
          method: options.method ?? 'GET',
          headers,
          body,
          signal: controller.signal,
        });
        const text = await response.text();
        let data = null;
        try {
          data = text ? JSON.parse(text) : null;
        } catch {
          data = text;
        }
        return {
          status: response.status,
          ok: response.ok,
          data,
        };
      } catch (error) {
        return {
          status: 502,
          ok: false,
          error: error instanceof Error ? error.message : String(error),
        };
      } finally {
        clearTimeout(timer);
      }
    },
  };
}
