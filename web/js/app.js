import { Grinder, selftest } from "./engine.js";
import { B58 } from "./ref.js";

const $ = (id) => document.getElementById(id);
const form = $("form"), startBtn = $("start"), stopBtn = $("stop"), hint = $("hint");
const screen = $("screen"), log = $("log"), progress = $("progress");
let grinder = null, running = false;
const session = [];   // every key found on this page: { address, secret, seed, pk }

const fmtN = (n) => n >= 1e12 ? (n / 1e12).toFixed(2) + "T" : n >= 1e9 ? (n / 1e9).toFixed(2) + "G" : n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : String(Math.round(n));
function fmtT(s) {
    if (!isFinite(s)) return "—";
    if (s < 1) return "<1s";
    if (s < 90) return s.toFixed(0) + "s";
    if (s < 5400) return (s / 60).toFixed(0) + "min";
    if (s < 172800) return (s / 3600).toFixed(1) + "h";
    if (s < 3.2e7) return (s / 86400).toFixed(1) + " days";
    return (s / 3.15e7).toFixed(1) + " years";
}

// ---------- terminal output ----------
const atBottom = () => screen.scrollHeight - screen.scrollTop - screen.clientHeight < 40;
function follow(stick) { if (stick) screen.scrollTop = screen.scrollHeight; }

function line(text, cls = "") {
    const stick = atBottom();
    const d = document.createElement("div");
    if (cls) d.className = cls;
    d.textContent = text || " ";
    log.appendChild(d);
    follow(stick);
    return d;
}
function echo(cmd) {
    if (log.childElementCount) line("");
    const d = line("");
    d.textContent = "";
    const ps = document.createElement("span");
    ps.className = "ps"; ps.textContent = "PS ~\\vanity> ";
    d.append(ps, document.createTextNode(cmd));
    screen.scrollTop = screen.scrollHeight;
}
function print(templateId) {
    log.appendChild(document.getElementById(templateId).content.cloneNode(true));
    screen.scrollTop = screen.scrollHeight;
}

// ---------- speed bars: RTX 5080 references + this device, if measured ----------
const MY_KEY = "vanity.myDevice";
const loadMine = () => { try { return JSON.parse(localStorage.getItem(MY_KEY)); } catch { return null; } };
function barRows() {
    const rows = [
        { label: "v1.0.0  native ", sub: "RTX 5080", v: 70e6, cls: "b1" },
        { label: "browser WebGPU ", sub: "RTX 5080", v: 145e6, cls: "b2" },
        { label: "v2.0.0  native ", sub: "RTX 5080", v: 410e6, cls: "b3" },
    ];
    const mine = loadMine();
    if (mine) rows.unshift({ label: "YOUR DEVICE    ", sub: mine.name, v: mine.rate, cls: "b4", mine: true });
    return rows;
}
function renderBars(slot) {
    const rows = barRows();
    const ch = (parseFloat(getComputedStyle(slot).fontSize) || 15) * 0.6;
    const cols = Math.floor(slot.clientWidth / ch) || 60;
    const wide = cols >= 64;                                   // room for the "RTX 5080" column
    const width = Math.max(6, Math.min(30, cols - 15 - 6 - (wide ? 13 : 1)));
    const max = Math.max(...rows.map((r) => r.v));
    const fmtV = (v) => (v >= 1e8 ? (v / 1e6).toFixed(0) : (v / 1e6).toFixed(1)) + "M";
    const pre = document.createElement("pre");
    pre.className = "bars";
    rows.forEach((r, i) => {
        const n = Math.max(1, Math.round((r.v / max) * width));
        const lab = document.createElement("span"); lab.className = r.mine ? "you" : "dim"; lab.textContent = r.label;
        const on = document.createElement("span"); on.className = r.cls; on.textContent = "█".repeat(n);
        const off = document.createElement("span"); off.className = "b0"; off.textContent = "█".repeat(width - n);
        const val = document.createElement("span"); val.className = r.mine ? "you" : "txt"; val.textContent = " " + fmtV(r.v).padStart(5);
        pre.append(lab, on, off, val);
        if (wide) {
            const sub = document.createElement("span"); sub.className = "dim"; sub.textContent = "  " + (r.mine ? "← " : "") + r.sub;
            pre.append(sub);
        }
        if (i < rows.length - 1) pre.append("\n");
    });
    const mine = rows.find((r) => r.mine);
    const out = [pre];
    if (mine && !wide) {
        const d = document.createElement("div"); d.className = "dim"; d.textContent = "YOUR DEVICE = " + mine.sub;
        out.push(d);
    }
    slot.replaceChildren(...out);
}
const refreshBars = () => document.querySelectorAll(".bars-slot").forEach(renderBars);
addEventListener("resize", refreshBars);

// ---------- commands ----------
const COMMANDS = {
    about: () => { echo("vanity --about"); print("t-about"); refreshBars(); },
    bench: () => runBench(),
    download: () => { echo("vanity --download"); print("t-download"); },
    demo: () => { echo("vanity --demo"); print("t-demo"); },
    how: () => { echo("cat HOW.md"); print("t-how"); },
    selftest: () => runSelftest(),
    save: () => {
        echo("vanity --save-all");
        if (!session.length) { line("nothing found yet", "dim"); return; }
        saveLine(session, "session");
    },
    clear: () => { if (!running) { log.replaceChildren(); progress.textContent = ""; } },
};
document.querySelectorAll("[data-cmd]").forEach((b) => b.addEventListener("click", () => COMMANDS[b.dataset.cmd]()));
// Alt+1..7 like the key bar
const KEYS = ["about", "download", "demo", "how", "selftest", "save", "clear"];
document.addEventListener("keydown", (e) => {
    if (!e.altKey || e.ctrlKey || e.metaKey) return;
    const n = Number(e.key);
    if (n >= 1 && n <= 7) { e.preventDefault(); COMMANDS[KEYS[n - 1]](); }
    else if (n === 8) { e.preventDefault(); location.href = "https://github.com/alhimikix/solana-vanity-gpu"; }
});

// "try:" links inside the about text fill the prompt and run it; buttons printed into the log run commands
log.addEventListener("click", (e) => {
    const c = e.target.closest("[data-cmd]");
    if (c) { e.preventDefault(); COMMANDS[c.dataset.cmd](); return; }
    const t = e.target.closest("[data-try]");
    if (!t) return;
    e.preventDefault();
    const [suf, pre, ic] = t.dataset.try.split("|");
    $("suffix").value = suf || ""; $("prefix").value = pre || ""; $("icase").checked = ic === "i";
    validate();
    if (!running) form.requestSubmit();
});

// ---------- prompt ----------
function readForm() {
    return { prefix: $("prefix").value.trim(), suffix: $("suffix").value.trim(), icase: $("icase").checked, count: Number($("count").value) };
}
function validate() {
    const { prefix, suffix, icase } = readForm();
    hint.classList.remove("err");
    if (!prefix && !suffix) { hint.textContent = "# type a prefix (-p), a suffix (-s), or both, then ⏎"; return false; }
    const bad = [...prefix + suffix].filter((c) => !B58.includes(c) && !(icase && (B58.includes(c.toLowerCase()) || B58.includes(c.toUpperCase()))));
    if (bad.length) {
        hint.classList.add("err");
        hint.textContent = `error: not in base58: ${[...new Set(bad)].join(" ")} — addresses never contain 0, O, I, l` + (icase ? "" : " (try -i)");
        return false;
    }
    const n = prefix.length + suffix.length;
    hint.textContent = n >= 7 ? "# 7+ characters can take hours in a browser — the native app is ~3x faster (see download)" : "# ⏎ to run";
    return true;
}
["prefix", "suffix", "icase"].forEach((id) => $(id).addEventListener("input", validate));

function act(label, fn) {
    const b = document.createElement("button");
    b.className = "act"; b.type = "button"; b.textContent = `[${label}]`;
    b.addEventListener("click", fn);
    return b;
}
async function copy(text, btn) {
    await navigator.clipboard.writeText(text);
    const old = btn.textContent;
    btn.textContent = "[copied]";
    setTimeout(() => (btn.textContent = old), 1200);
}

// ---------- export ----------
function download(name, text, type) {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const el = document.createElement("a");
    el.href = url; el.download = name; el.click();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
}
function toCsv(list) {   // same format as the native app
    return "public_key,private_key\n" + list.map((m) => `${m.address},${m.secret}`).join("\n") + "\n";
}
function toJson(list) {
    return JSON.stringify(list.map((m) => ({ publicKey: m.address, privateKey: m.secret, keypair: [...m.seed, ...m.pk] })), null, 1);
}
function saveLine(list, tag) {
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    const d = line("");
    d.textContent = "";
    const t = document.createElement("span");
    t.className = "dim"; t.textContent = `save ${list.length} keypair${list.length > 1 ? "s" : ""}: `;
    d.append(t,
        act("csv", () => download(`vanity-${tag}-${stamp}.csv`, toCsv(list), "text/csv")),
        act("json", () => download(`vanity-${tag}-${stamp}.json`, toJson(list), "application/json")));
}

function addFound(m, n, prefix, suffix) {
    const stick = atBottom();
    const row = document.createElement("div");
    const head = document.createElement("span");
    head.className = "found"; head.textContent = `Found #${n}: `;
    const addr = document.createElement("span");
    const a = m.address, p = prefix.length, s = suffix.length;
    for (const [t, hl] of [[a.slice(0, p), true], [a.slice(p, a.length - s), false], [a.slice(a.length - s), true]]) {
        if (!t) continue;
        const el = document.createElement(hl ? "mark" : "span");
        el.textContent = t;
        addr.appendChild(el);
    }
    const secret = document.createElement("div");
    secret.className = "secret hidden";
    secret.textContent = "private key: " + m.secret;
    const acts = document.createElement("span");
    acts.className = "acts";
    acts.append(
        act("copy address", (e) => copy(m.address, e.target)),
        act("copy key", (e) => copy(m.secret, e.target)),
        act("show key", (e) => { secret.classList.toggle("hidden"); e.target.textContent = secret.classList.contains("hidden") ? "[show key]" : "[hide key]"; }),
        act("keypair.json", () => {
            const url = URL.createObjectURL(new Blob([m.json], { type: "application/json" }));
            const el = document.createElement("a");
            el.href = url; el.download = `${m.address}.json`; el.click();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        }));
    row.append(head, addr, acts);
    log.append(row, secret);
    follow(stick);
}

async function getGrinder() {
    if (grinder) return grinder;
    line("compiling GPU kernels (first run only, up to ~20s)…", "dim");
    const t = performance.now();
    grinder = await Grinder.create({ W: 14, logNK: 20, mc: 64 });
    const gpuName = [grinder.info.vendor, grinder.info.architecture, grinder.info.description].filter(Boolean).join(" ") || "GPU";
    grinder.gpuName = gpuName;
    $("gputitle").textContent = `solana-vanity-gpu v2.0.0 — ~\\vanity — WebGPU: ${gpuName}`;
    line(`GPU:      ${gpuName} (WebGPU)`, "dim");
    line(`          comb W=${grinder.cfg.w}: ${grinder.cfg.nwin} windows, table ${(grinder.cfg.entries * 96 / 1048576).toFixed(1)} MB, ready in ${((performance.now() - t) / 1000).toFixed(1)}s`, "dim");
    return grinder;
}

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (running || !validate()) return;
    const { prefix, suffix, icase, count } = readForm();
    running = true;
    startBtn.disabled = true; stopBtn.disabled = false;
    echo(["vanity", prefix && `-p ${prefix}`, suffix && `-s ${suffix}`, icase && "-i", count && `-n ${count}`].filter(Boolean).join(" "));
    let timer = null;
    try {
        const g = await getGrinder();
        const pats = g.setPatterns([{ prefix, suffix }], icase);
        line(`Patterns: ${prefix}*${suffix}${icase ? "  (case-insensitive)" : ""}`);
        line(`Odds:     1 in ${(1 / pats.prob).toPrecision(4)} keys per match`, "y");
        line("");
        let got = 0, sinceFind = 0, prevKeys = 0, last = null, frame = 0;
        const run = [];
        const t0 = performance.now();
        const SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";
        timer = setInterval(() => {
            if (!last) return;
            const exp = 1 / pats.prob, n = Math.round(Math.min(1, sinceFind / exp) * 24);
            const sp = document.createElement("span"); sp.className = "spin"; sp.textContent = SPIN[frame++ % SPIN.length] + " ";
            const a = document.createElement("span"); a.className = "pb"; a.textContent = "█".repeat(n);
            const b = document.createElement("span"); b.className = "pb0"; b.textContent = "█".repeat(24 - n);
            progress.replaceChildren(sp, a, b, document.createTextNode(
                ` ${(100 * sinceFind / exp).toFixed(0).padStart(3)}% of expected | tried ${fmtN(last.keys)} | ${fmtN(last.rate)} keys/sec | found ${got} | ETA/match ${fmtT(exp / last.avg)}`));
        }, 100);
        await g.run({
            onMatch: (m) => {
                if (count && got >= count) return;
                got++; sinceFind = 0;
                run.push(m); session.push(m);
                addFound(m, got, prefix, suffix);
                if (count && got >= count) g.stop();
            },
            onStats: (s) => { sinceFind += s.keys - prevKeys; prevKeys = s.keys; last = s; },
            onError: (err) => line("error: " + (err.message || err), "r"),
        });
        clearInterval(timer);
        progress.textContent = "";
        const secs = (performance.now() - t0) / 1000;
        // very short runs are dominated by start-up latency, so the average would understate the speed
        line(`Done. ${got} keypairs in ${secs.toFixed(2)}s` + (secs > 3 && last ? ` (${fmtN(last.avg)} keys/sec average).` : "."), "done");
        if (run.length) saveLine(run, [prefix, suffix].filter(Boolean).join("-"));
    } catch (err) {
        if (timer) clearInterval(timer);
        progress.textContent = "";
        line("error: " + (err.message || err), "r");
    }
    running = false;
    startBtn.disabled = !navigator.gpu; stopBtn.disabled = true;
});

stopBtn.addEventListener("click", () => { grinder?.stop(); });

// Measure this device: ~10 s on a pattern that practically never matches.
async function runBench() {
    if (running) return;
    if (!navigator.gpu) { echo("vanity --bench"); line("no WebGPU in this browser — try Chrome or Edge", "r"); return; }
    running = true; startBtn.disabled = true;
    echo("vanity --bench");
    let timer = null;
    try {
        const g = await getGrinder();
        g.setPatterns([{ suffix: "zzzzzzzzzz" }], false);
        const windows = [], SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", t0 = performance.now(), SECS = 10;
        let frame = 0, last = null;
        timer = setInterval(() => {
            const el = (performance.now() - t0) / 1000;
            progress.textContent = `${SPIN[frame++ % SPIN.length]} measuring ${Math.min(SECS, el).toFixed(0)}/${SECS}s` + (last ? ` | ${fmtN(last.rate)} keys/sec` : "");
        }, 100);
        setTimeout(() => g.stop(), SECS * 1000);
        await g.run({ onStats: (s) => { last = s; if (performance.now() - t0 > 3000) windows.push(s.rate); }, onError: (err) => line("error: " + (err.message || err), "r") });
        clearInterval(timer); timer = null;
        progress.textContent = "";
        // steady state: median of the windows after the batch size settled
        windows.sort((a, b) => a - b);
        const rate = windows.length ? windows[Math.floor(windows.length / 2)] : (last ? last.avg : 0);
        if (!rate) throw new Error("no measurement");
        localStorage.setItem(MY_KEY, JSON.stringify({ rate, name: g.gpuName + " (this browser)" }));
        line(`your device: ${g.gpuName}, WebGPU in this browser`);
        const d = line(""); d.textContent = "";
        const sp = document.createElement("span"); sp.className = "you"; sp.textContent = `speed:       ${fmtN(rate)} keys/sec`;
        d.append(sp);
        const slot = document.createElement("div"); slot.className = "bars-slot";
        log.appendChild(slot); renderBars(slot);
        line("");
        line("average wait for one address on this device:", "dim");
        const rowsT = [[3, "abc"], [4, "pump"], [5, "Anton"], [6, "Solana"], [7, "Phantom"], [8, "Solana42"]];
        for (const [k, ex] of rowsT) line(`  ${k} chars ${("(" + ex + ")").padEnd(11)} ${fmtT(58 ** k / rate).padStart(10)}`, k <= 5 ? "" : "dim");
        line("  a prefix starting with J…z takes ~17× longer", "dim");
        line("  ignoring case (-i) is several times faster", "dim");
        if (rate < 100e6) line("  the native app on an NVIDIA GPU reaches ~410M keys/sec — see [2] download app", "dim");
        refreshBars();
    } catch (err) {
        if (timer) clearInterval(timer);
        progress.textContent = "";
        line("bench error: " + (err.message || err), "r");
    }
    running = false; startBtn.disabled = !navigator.gpu;
}

async function runSelftest() {
    if (running) return;
    running = true; startBtn.disabled = true;
    echo("vanity --selftest");
    line("running (about half a minute)…", "dim");
    try {
        const r = await selftest({ W: 14 });
        for (const l of r.lines) line("[" + (l.startsWith("ok") ? " ok " : "FAIL") + "] " + l.slice(5), l.startsWith("ok") ? "" : "r");
        line(r.ok ? "SELFTEST PASSED" : "SELFTEST FAILED", r.ok ? "done" : "r");
    } catch (err) { line("selftest error: " + (err.message || err), "r"); }
    running = false; startBtn.disabled = !navigator.gpu;
}

// ---------- start ----------
COMMANDS.about();
if (!navigator.gpu) {
    line("");
    line("This browser has no WebGPU. Use a recent Chrome or Edge, or get the native app:", "r");
    COMMANDS.download();
    startBtn.disabled = true;
}
validate();
$("prefix").focus();
