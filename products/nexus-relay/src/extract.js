const ENTITY_RE = /&(#\d+|#x[0-9a-f]+|amp|lt|gt|quot|apos);/gi;

export function decodeEntities(value = "") {
  return value.replace(ENTITY_RE, (match, entity) => {
    const key = entity.toLowerCase();
    if (key === "amp") return "&";
    if (key === "lt") return "<";
    if (key === "gt") return ">";
    if (key === "quot") return '"';
    if (key === "apos") return "'";
    if (key.startsWith("#x")) return String.fromCodePoint(parseInt(key.slice(2), 16));
    if (key.startsWith("#")) return String.fromCodePoint(parseInt(key.slice(1), 10));
    return match;
  });
}

export function stripHtml(html = "") {
  return decodeEntities(
    html
      .replace(/<!--[\s\S]*?-->/g, " ")
      .replace(/<script\b[\s\S]*?<\/script>/gi, " ")
      .replace(/<style\b[\s\S]*?<\/style>/gi, " ")
      .replace(/<noscript\b[\s\S]*?<\/noscript>/gi, " ")
      .replace(/<svg\b[\s\S]*?<\/svg>/gi, " ")
      .replace(/<[^>]+>/g, " ")
      .replace(/\s+/g, " ")
      .trim()
  );
}

function firstMatch(html, patterns) {
  for (const pattern of patterns) {
    const match = html.match(pattern);
    if (match?.[1]) return decodeEntities(match[1].trim());
  }
  return "";
}

export function extractMeta(html = "", url = "") {
  const title = firstMatch(html, [
    /<meta[^>]+property=["']og:title["'][^>]+content=["']([^"']+)["']/i,
    /<meta[^>]+content=["']([^"']+)["'][^>]+property=["']og:title["']/i,
    /<title[^>]*>([\s\S]*?)<\/title>/i
  ]);
  const description = firstMatch(html, [
    /<meta[^>]+name=["']description["'][^>]+content=["']([^"']+)["']/i,
    /<meta[^>]+property=["']og:description["'][^>]+content=["']([^"']+)["']/i,
    /<meta[^>]+content=["']([^"']+)["'][^>]+name=["']description["']/i
  ]);
  const canonical = firstMatch(html, [
    /<link[^>]+rel=["']canonical["'][^>]+href=["']([^"']+)["']/i,
    /<link[^>]+href=["']([^"']+)["'][^>]+rel=["']canonical["']/i
  ]);
  return { title, description, canonical_url: canonical || url };
}

export function extractJsonLd(html = "") {
  const out = [];
  const pattern = /<script[^>]+type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi;
  let match;
  while ((match = pattern.exec(html)) !== null && out.length < 8) {
    try { out.push(JSON.parse(match[1].trim())); } catch { /* malformed JSON-LD */ }
  }
  return out;
}

export function normalizeHtml({ html, url, fetchedAt, status, headers }) {
  const meta = extractMeta(html, url);
  const text = stripHtml(html);
  return {
    schema: "nexus.relay.document.v1",
    kind: "document",
    source: { url, canonical_url: meta.canonical_url, fetched_at: fetchedAt, status },
    content: { title: meta.title, description: meta.description, text, text_length: text.length, json_ld: extractJsonLd(html) },
    transport: { content_type: headers.get("content-type") || "text/html", etag: headers.get("etag"), last_modified: headers.get("last-modified") }
  };
}

function tagValue(block, tag) {
  const match = block.match(new RegExp(`<${tag}\\b[^>]*>([\\s\\S]*?)<\\/${tag}>`, "i"));
  return match ? stripHtml(match[1]) : "";
}

export function normalizeFeed({ xml, url, fetchedAt, status, headers }) {
  const blocks = [...xml.matchAll(/<(item|entry)\b[\s\S]*?<\/\1>/gi)].slice(0, 100);
  const items = blocks.map((match, index) => {
    const block = match[0];
    const href = block.match(/<link[^>]+href=["']([^"']+)["']/i)?.[1] || tagValue(block, "link");
    return {
      id: tagValue(block, "guid") || tagValue(block, "id") || href || `${url}#${index}`,
      title: tagValue(block, "title"),
      url: href,
      published_at: tagValue(block, "pubDate") || tagValue(block, "published") || tagValue(block, "updated"),
      author: tagValue(block, "author") || tagValue(block, "creator"),
      text: tagValue(block, "description") || tagValue(block, "summary") || tagValue(block, "content")
    };
  });
  return {
    schema: "nexus.relay.feed.v1",
    kind: "feed",
    source: { url, fetched_at: fetchedAt, status },
    content: { title: tagValue(xml, "title"), items, item_count: items.length },
    transport: { content_type: headers.get("content-type") || "application/xml" }
  };
}

export function normalizeJson({ value, url, fetchedAt, status, headers }) {
  return {
    schema: "nexus.relay.json.v1",
    kind: "json",
    source: { url, fetched_at: fetchedAt, status },
    content: value,
    transport: { content_type: headers.get("content-type") || "application/json" }
  };
}

export function normalizeText({ text, url, fetchedAt, status, headers, kind = "text" }) {
  return {
    schema: `nexus.relay.${kind}.v1`,
    kind,
    source: { url, fetched_at: fetchedAt, status },
    content: { text, text_length: text.length },
    transport: { content_type: headers.get("content-type") || "text/plain" }
  };
}

function parseCsvLine(line) {
  const cells = [];
  let value = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"' && quoted && line[i + 1] === '"') { value += '"'; i += 1; }
    else if (ch === '"') quoted = !quoted;
    else if (ch === "," && !quoted) { cells.push(value); value = ""; }
    else value += ch;
  }
  cells.push(value);
  return cells;
}

export function normalizeCsv({ text, url, fetchedAt, status, headers }) {
  const lines = text.split(/\r?\n/).filter((line) => line.length).slice(0, 1001);
  const header = lines.length ? parseCsvLine(lines[0]) : [];
  const rows = lines.slice(1).map((line) => {
    const cells = parseCsvLine(line);
    return Object.fromEntries(header.map((name, index) => [name || `column_${index + 1}`, cells[index] ?? ""]));
  });
  return {
    schema: "nexus.relay.csv.v1",
    kind: "csv",
    source: { url, fetched_at: fetchedAt, status },
    content: { columns: header, rows, row_count: rows.length, truncated: text.split(/\r?\n/).length > 1001 },
    transport: { content_type: headers.get("content-type") || "text/csv" }
  };
}

export function normalizeSitemap({ xml, url, fetchedAt, status, headers }) {
  const entries = [...xml.matchAll(/<url\b[\s\S]*?<\/url>/gi)].slice(0, 1000).map((match) => ({
    url: tagValue(match[0], "loc"),
    last_modified: tagValue(match[0], "lastmod"),
    change_frequency: tagValue(match[0], "changefreq"),
    priority: tagValue(match[0], "priority")
  })).filter((entry) => entry.url);
  const indexes = [...xml.matchAll(/<sitemap\b[\s\S]*?<\/sitemap>/gi)].slice(0, 1000).map((match) => ({
    url: tagValue(match[0], "loc"),
    last_modified: tagValue(match[0], "lastmod")
  })).filter((entry) => entry.url);
  return {
    schema: "nexus.relay.sitemap.v1",
    kind: "sitemap",
    source: { url, fetched_at: fetchedAt, status },
    content: { urls: entries, sitemap_indexes: indexes, item_count: entries.length + indexes.length },
    transport: { content_type: headers.get("content-type") || "application/xml" }
  };
}
