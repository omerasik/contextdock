const el = id => document.getElementById(id);
let token = "";
let editing = null;
let signedIn = 0;
let busy = false;

function node(tag, text) {
  const result = document.createElement(tag);
  result.textContent = text;
  return result;
}

function button(text, handler) {
  const result = node("button", text);
  result.type = "button";
  result.onclick = handler;
  return result;
}

async function api(route, method = "GET", body) {
  const response = await fetch("/api/" + route, {
    method, headers: { Authorization: "Bearer " + token, ...(body ? { "Content-Type": "application/json" } : {}) },
    body: body ? JSON.stringify(body) : undefined
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail.replaceAll("_", " ") : "Check the submitted fields.");
  return payload;
}

async function action(callback) {
  if (busy) return;
  busy = true;
  try { el("notice").textContent = ""; await callback(); }
  catch (error) { el("notice").textContent = error.message; }
  finally { busy = false; }
}

function onSubmit(id, callback) {
  const form = el(id);
  form.onsubmit = event => {
    event.preventDefault();
    const values = Object.fromEntries(new FormData(form));
    void action(() => callback(values));
  };
}

async function showSource(id, ref) {
  const source = await api("documents/" + id);
  const text = ref ? Array.from(source.body).slice(ref.start, ref.end).join("") : source.body;
  el("source").replaceChildren(node("h3", source.title), node("pre", text));
}

async function refresh() {
  const epoch = signedIn;
  const [settings, docs, saved, audit] = await Promise.all([api("settings"), api("documents"), api("memory"), api("audit")]);
  if (epoch !== signedIn) return;
  el("tenant").textContent = settings.tenant;
  el("settings").elements.retention_days.value = settings.retention_days;
  el("documents").replaceChildren(...docs.map(doc => {
    const row = node("div", ""); row.className = "record";
    row.append(node("p", doc.collection + " · " + doc.title),
      button("Source", () => void action(() => showSource(doc.id))),
      button("Delete", () => void action(async () => {
        if (!window.confirm("Delete this document and its indexed chunks?")) return;
        await api("documents/" + doc.id, "DELETE");
        el("results").replaceChildren(); el("source").replaceChildren(); await refresh();
      })));
    return row;
  }));
  el("memories").replaceChildren(...saved.map(item => {
    const row = node("div", ""); row.className = "record";
    row.append(node("p", item.name + " · " + item.value), button("Edit", () => {
      editing = item;
      el("remember").elements.name.value = item.name;
      el("remember").elements.value.value = item.value;
      el("remember").elements.value.focus();
    }), button("Delete", () => void action(async () => {
      await api("memory/" + item.name, "PUT", { value: null, expected: item.version });
      await refresh();
    })));
    return row;
  }));
  el("audit").replaceChildren(...audit.slice(0, 10).map(item => node("li", item.action.replaceAll("_", " "))));
}

async function connect(value) {
  token = value;
  signedIn += 1;
  try { await refresh(); }
  catch (error) { token = ""; throw error; }
  el("workspace").hidden = false;
  el("connect").hidden = true;
  el("logout").hidden = false;
}

onSubmit("connect", data => { el("connect").reset(); return connect(data.token); });
el("logout").onclick = () => {
  if (busy) return;
  signedIn += 1;
  token = "";
  editing = null;
  el("tenant").textContent = "";
  el("workspace").hidden = true;
  el("connect").hidden = false;
  el("logout").hidden = true;
  for (const id of ["results", "source", "documents", "memories", "audit", "scores"]) el(id).replaceChildren();
  for (const id of ["ingest", "search", "remember", "settings"]) el(id).reset();
};
onSubmit("ingest", async data => {
  const result = await api("documents", "POST", data);
  await refresh();
  el("notice").textContent = result.duplicate ? "This content was already indexed in the collection." : "Indexed " + result.chunks + " chunks.";
});
el("file").onchange = () => void action(async () => {
  const file = el("file").files[0];
  if (!file) return;
  if (file.size > 72_000) throw new Error("Text uploads must fit the 18,000 character limit.");
  const text = new TextDecoder("utf-8", { fatal: true }).decode(await file.arrayBuffer());
  if (text.length > 18_000) throw new Error("Text uploads must fit the 18,000 character limit.");
  el("ingest").elements.text.value = text;
  if (!el("ingest").elements.title.value) el("ingest").elements.title.value = file.name.slice(0, 120);
});
onSubmit("search", async data => {
  const result = await api("search", "POST", { query: data.query, collection: data.collection || null, threshold: Number(data.threshold) });
  await refresh();
  const results = el("results");
  el("source").replaceChildren();
  results.replaceChildren(node("h3", "Grounded excerpts"));
  if (!result.answer.length) results.append(node("p", "No matching excerpt was found."));
  for (const answer of result.answer) {
    results.append(node("p", answer.quote), button("View cited text", () => void action(() => showSource(answer.document_id, answer))));
  }
  for (const source of result.sources) {
    const row = node("div", ""); row.className = "record";
    row.append(node("p", source.title + " · score " + source.score), button("View matching chunk", () => void action(() => showSource(source.document_id, source))));
    results.append(row);
  }
});
onSubmit("remember", async data => {
  await api("memory/" + data.name, "PUT", { value: data.value, expected: editing?.name === data.name ? editing.version : null });
  editing = null;
  el("remember").reset();
  await refresh();
});
el("remember").elements.name.oninput = () => { editing = null; };
onSubmit("settings", async data => {
  await api("settings", "PUT", { retention_days: Number(data.retention_days) });
  await refresh();
});
const config = await api("config");
if (config.demo) {
  el("demo").hidden = false;
  el("evaluation").hidden = false;
  for (const control of el("demo").querySelectorAll("[data-user]")) {
    control.onclick = () => void action(() => connect("contextdock-demo-" + control.dataset.user));
  }
  el("evaluate").onclick = () => void action(async () => { el("scores").textContent = JSON.stringify(await api("evaluation"), null, 2); });
}
