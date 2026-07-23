export type NativeEnv = {
  AURO_NATIVE_BASE_URL: string;
  AURO_NATIVE_API_TOKEN?: string;
  AURO_NATIVE_MODEL?: string;
};

function endpoint(env: NativeEnv, path: string): string {
  const base = env.AURO_NATIVE_BASE_URL?.replace(/\/$/, "");
  if (!base) throw new Error("AURO_NATIVE_BASE_URL is not configured");
  return `${base}${path}`;
}

function headers(env: NativeEnv): HeadersInit {
  const value: Record<string, string> = { "content-type": "application/json", "x-auro-commercial-gateway": "1" };
  if (env.AURO_NATIVE_API_TOKEN) value.authorization = `Bearer ${env.AURO_NATIVE_API_TOKEN}`;
  return value;
}

export async function nativeGet(env: NativeEnv, path: string): Promise<Response> {
  return fetch(endpoint(env, path), { headers: headers(env), cf: { cacheTtl: 0, cacheEverything: false } });
}

export async function nativeChat(env: NativeEnv, messages: unknown[], maxTokens: number): Promise<Response> {
  return fetch(endpoint(env, "/v1/chat/completions"), {
    method: "POST",
    headers: headers(env),
    body: JSON.stringify({
      model: env.AURO_NATIVE_MODEL || "auro-him",
      messages,
      max_tokens: maxTokens,
      stream: false
    })
  });
}

export async function relay(response: Response): Promise<Response> {
  const outgoing = new Headers(response.headers);
  outgoing.set("cache-control", "no-store");
  outgoing.set("x-auro-native", "true");
  outgoing.delete("set-cookie");
  return new Response(response.body, { status: response.status, headers: outgoing });
}
