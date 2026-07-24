/**
 * Auro Pythonista host — JS service that renders Python-declared UIs + DB tables.
 *
 * Pattern (Pythonista app + succotash AppServiceBridge):
 *   Python scripts run in background (server)
 *   They present(ui.View) + write DB rows
 *   This host paints the UI tree as DOM and tables as data grids
 *   Button taps POST back as /v1/pythonista/event
 */
(function (global) {
  const API = (global.AURO_API_BASE || "");

  async function api(path, body) {
    const r = await fetch(API + path, {
      method: body ? "POST" : "GET",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    return r.json();
  }

  function el(tag, attrs, children) {
    const n = document.createElement(tag);
    if (attrs) {
      Object.entries(attrs).forEach(([k, v]) => {
        if (k === "style" && typeof v === "object") Object.assign(n.style, v);
        else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
        else if (v != null) n.setAttribute(k, v);
      });
    }
    (children || []).forEach((c) => {
      if (c == null) return;
      n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return n;
  }

  function renderNode(node, host) {
    if (!node) return el("div", { class: "py-empty" }, ["(empty ui)"]);
    const type = node.type || "View";
    const props = node.props || {};
    const bg = node.background_color || "transparent";
    const wrapStyle = {
      background: bg,
      padding: "8px",
      borderRadius: "8px",
      marginBottom: "6px",
      border: "1px solid #243049",
    };

    if (type === "Label") {
      return el(
        "div",
        {
          class: "py-label",
          style: {
            color: props.text_color || "#e8eefc",
            fontSize: (props.font_size || 14) + "px",
            textAlign: props.alignment || "left",
            margin: "4px 0",
          },
        },
        [props.text || ""]
      );
    }

    if (type === "Button") {
      return el(
        "button",
        {
          class: "py-btn",
          style: {
            background: props.tint_color || "#5b8cff",
            color: "#fff",
            border: "0",
            borderRadius: "8px",
            padding: "8px 12px",
            cursor: "pointer",
            margin: "4px 0",
            width: "100%",
            fontWeight: "600",
          },
          onclick: async () => {
            const field = host.querySelector('input[data-py-name="title_field"]');
            const payload = { text: field ? field.value : "" };
            const res = await api("/v1/pythonista/event", {
              node_id: node.id,
              action: "action",
              payload,
            });
            if (res.ui) paintUI(host, res.ui);
            if (res.tables) paintTables(host, res.tables);
            host.dispatchEvent(new CustomEvent("pythonista:event", { detail: res }));
          },
        },
        [props.title || "Button"]
      );
    }

    if (type === "TextField") {
      return el("input", {
        class: "py-field",
        type: "text",
        placeholder: props.placeholder || "",
        value: props.text || "",
        "data-py-name": node.name || "",
        style: {
          width: "100%",
          background: "#0d1424",
          color: "#e8eefc",
          border: "1px solid #243049",
          borderRadius: "8px",
          padding: "8px",
          margin: "4px 0",
        },
      });
    }

    if (type === "TextView") {
      return el(
        "textarea",
        {
          class: "py-textview",
          style: {
            width: "100%",
            minHeight: "80px",
            background: "#0d1424",
            color: "#e8eefc",
            border: "1px solid #243049",
            borderRadius: "8px",
            padding: "8px",
          },
        },
        [props.text || ""]
      );
    }

    if (type === "TableView") {
      return renderTable({
        columns: props.columns || [],
        rows: props.rows || [],
        table: props.source_table || node.name || "table",
      });
    }

    if (type === "WebView") {
      if (props.url) {
        return el("iframe", {
          src: props.url,
          style: { width: "100%", minHeight: "200px", border: "1px solid #243049", borderRadius: "8px" },
        });
      }
      const frame = el("div", {
        class: "py-webview",
        style: { padding: "8px", background: "#0d1424", borderRadius: "8px" },
      });
      frame.innerHTML = props.html || "";
      return frame;
    }

    // StackView / View
    const axis = props.axis === "horizontal" ? "row" : "column";
    const box = el("div", {
      class: "py-" + type.toLowerCase(),
      style: {
        ...wrapStyle,
        display: "flex",
        flexDirection: axis,
        gap: (props.spacing || 8) + "px",
      },
    });
    (node.children || []).forEach((c) => box.appendChild(renderNode(c, host)));
    return box;
  }

  function renderTable(spec) {
    const columns = spec.columns && spec.columns.length
      ? spec.columns
      : (spec.rows && spec.rows[0] ? Object.keys(spec.rows[0]) : []);
    const table = el("table", {
      class: "py-table",
      style: {
        width: "100%",
        borderCollapse: "collapse",
        fontSize: "12px",
        marginTop: "6px",
      },
    });
    const thead = el("thead");
    const hr = el("tr");
    columns.forEach((c) =>
      hr.appendChild(
        el(
          "th",
          {
            style: {
              textAlign: "left",
              padding: "6px",
              borderBottom: "1px solid #243049",
              color: "#8b9bb8",
            },
          },
          [String(c)]
        )
      )
    );
    thead.appendChild(hr);
    table.appendChild(thead);
    const tbody = el("tbody");
    (spec.rows || []).forEach((row) => {
      const tr = el("tr");
      columns.forEach((c) =>
        tr.appendChild(
          el(
            "td",
            {
              style: {
                padding: "6px",
                borderBottom: "1px solid #1a2236",
                color: "#e8eefc",
                maxWidth: "220px",
                overflow: "hidden",
                textOverflow: "ellipsis",
              },
            },
            [row[c] == null ? "" : String(row[c])]
          )
        )
      );
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return el("div", { class: "py-table-wrap" }, [
      el("div", { style: { fontSize: "11px", color: "#8b9bb8", marginBottom: "4px" } }, [
        "DB / " + (spec.table || "table") + " · " + ((spec.rows && spec.rows.length) || 0) + " rows",
      ]),
      table,
    ]);
  }

  function paintUI(host, tree) {
    host.innerHTML = "";
    const root = tree && tree.root ? tree.root : tree;
    if (!root || tree.type === "empty") {
      host.appendChild(el("div", { style: { color: "#8b9bb8" } }, ["No UI presented yet. Run a Python script."]));
      return;
    }
    host.appendChild(renderNode(root, host));
  }

  function paintTables(host, tables) {
    let panel = host.querySelector(".py-db-panel");
    if (!panel) {
      panel = el("div", { class: "py-db-panel", style: { marginTop: "12px" } });
      host.appendChild(panel);
    }
    panel.innerHTML = "";
    panel.appendChild(el("div", { style: { color: "#8b9bb8", fontSize: "12px", marginBottom: "6px" } }, ["Database tables"]));
    (tables || []).forEach((t) => {
      if (t && t.ok !== false) panel.appendChild(renderTable(t));
    });
  }

  async function runScript(opts) {
    return api("/v1/pythonista/run", opts || {});
  }

  async function mount(container, options) {
    const host = typeof container === "string" ? document.querySelector(container) : container;
    if (!host) throw new Error("pythonista host container missing");
    options = options || {};

    const chrome = el("div", { class: "pyista-chrome", style: { display: "flex", flexDirection: "column", gap: "8px", height: "100%" } });
    const toolbar = el("div", { style: { display: "flex", gap: "6px", flexWrap: "wrap" } });
    const canvas = el("div", {
      class: "pyista-canvas",
      style: {
        flex: "1",
        overflow: "auto",
        background: "#0b1020",
        border: "1px solid #243049",
        borderRadius: "10px",
        padding: "10px",
        minHeight: "220px",
      },
    });
    const log = el("pre", {
      style: {
        maxHeight: "120px",
        overflow: "auto",
        fontSize: "11px",
        background: "#0d1424",
        border: "1px solid #243049",
        borderRadius: "8px",
        padding: "8px",
        color: "#8b9bb8",
      },
    });

    function btn(label, fn) {
      return el(
        "button",
        {
          style: {
            background: "#5b8cff",
            color: "#fff",
            border: 0,
            borderRadius: "8px",
            padding: "6px 10px",
            cursor: "pointer",
            fontWeight: 600,
          },
          onclick: fn,
        },
        [label]
      );
    }

    toolbar.appendChild(
      btn("Run dashboard.py", async () => {
        const res = await runScript({ script_name: "hello_dashboard.py" });
        log.textContent = JSON.stringify({ ok: res.ok, stdout: res.stdout, error: res.error }, null, 2);
        if (res.ui) paintUI(canvas, res.ui);
        if (res.tables) paintTables(canvas, res.tables);
      })
    );
    toolbar.appendChild(
      btn("BG pulse", async () => {
        const res = await runScript({ script_name: "bg_pulse.py", background: true, interval_s: 5 });
        log.textContent = JSON.stringify(res, null, 2);
      })
    );
    toolbar.appendChild(
      btn("Refresh UI/DB", async () => {
        const ui = await api("/v1/pythonista/ui");
        const tables = await api("/v1/pythonista/tables");
        paintUI(canvas, ui);
        paintTables(canvas, tables.renders || []);
        log.textContent = "refreshed " + new Date().toISOString();
      })
    );
    toolbar.appendChild(
      btn("Status", async () => {
        log.textContent = JSON.stringify(await api("/v1/pythonista/status"), null, 2);
      })
    );

    chrome.appendChild(toolbar);
    chrome.appendChild(canvas);
    chrome.appendChild(log);
    host.innerHTML = "";
    host.appendChild(chrome);

    // auto-run example if requested
    if (options.autorun) {
      const res = await runScript({ script_name: options.autorun });
      if (res.ui) paintUI(canvas, res.ui);
      if (res.tables) paintTables(canvas, res.tables);
      log.textContent = res.stdout || JSON.stringify(res, null, 2);
    }

    return { host, canvas, runScript, paintUI, paintTables, api };
  }

  global.AuroPythonista = { mount, runScript, paintUI, paintTables, api, renderTable };
})(typeof window !== "undefined" ? window : globalThis);
