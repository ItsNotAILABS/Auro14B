export const MCP_TOOLS = [
  {
    name: "relay_read",
    description: "Read a public URL through the hosted NEXUS Relay API and return normalized, provenance-bearing content with a billable usage receipt.",
    inputSchema: {
      type: "object",
      required: ["url"],
      properties: {
        url: { type: "string", format: "uri" },
        mode: { type: "string", enum: ["auto", "html", "feed", "json", "markdown", "text", "csv", "sitemap"], default: "auto" },
        max_bytes: { type: "integer", minimum: 1000, maximum: 5000000 },
        cache_ttl: { type: "integer", minimum: 60, maximum: 86400 }
      },
      additionalProperties: false
    }
  }
];

export function mcpResponse(id, result) {
  return { jsonrpc: "2.0", id, result };
}

export function mcpError(id, code, message, data) {
  return { jsonrpc: "2.0", id, error: { code, message, ...(data ? { data } : {}) } };
}

export async function handleMcp(request, readFn) {
  let body;
  try {
    body = await request.json();
  } catch {
    return mcpError(null, -32700, "Parse error");
  }
  const { id = null, method, params = {} } = body || {};
  if (method === "initialize") {
    return mcpResponse(id, {
      protocolVersion: "2025-06-18",
      capabilities: { tools: { listChanged: false } },
      serverInfo: { name: "nexus-relay", version: "0.2.0" }
    });
  }
  if (method === "notifications/initialized") return null;
  if (method === "tools/list") return mcpResponse(id, { tools: MCP_TOOLS });
  if (method === "tools/call") {
    if (params?.name !== "relay_read") return mcpError(id, -32602, "Unknown tool");
    try {
      const result = await readFn(params.arguments || {});
      return mcpResponse(id, {
        content: [{ type: "text", text: JSON.stringify(result) }],
        structuredContent: result,
        isError: false
      });
    } catch (error) {
      return mcpResponse(id, {
        content: [{ type: "text", text: String(error?.message || error) }],
        isError: true
      });
    }
  }
  return mcpError(id, -32601, "Method not found");
}
