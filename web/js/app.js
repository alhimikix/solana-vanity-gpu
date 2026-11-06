import { Grinder, selftest } from "./engine.js";
import { B58 } from "./ref.js";

const $ = (id) => document.getElementById(id);
const form = $("form"), startBtn = $("start"), stopBtn = $("stop"), hint = $("hint");
const screen = $("screen"), log = $("log"), progress = $("progress");
let grinder = null, running = false;

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

// ---------- commands from the title bar ----------
const COMMANDS = {
    about: () => { echo("vanity --about"); print("t-about"); },
    download: () => { echo("vanity --download"); print("t-download"); },
    demo: () => { echo("vanity --demo"); print("t-demo"); },
    how: () => { echo("cat HOW.md"); print("t-how"); },
    selftest: () => runSelftest(),
    clear: () => { if (!running) { log.replaceChildren(); progress.textContent = ""; } },
};
document.querySelectorAll("[data-cmd]").forEach((b) => b.addEventListener("click", () => COMMANDS[b.dataset.cmd]()));

// "try:" links inside the about text fill the prompt and run it
log.addEventListener("click", (e) => {
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
                addFound(m, got, prefix, suffix);
                if (count && got >= count) g.stop();
            },
            onStats: (s) => { sinceFind += s.keys - prevKeys; prevKeys = s.keys; last = s; },
            onError: (err) => line("error: " + (err.message || err), "r"),
        });
        clearInterval(timer);
        progress.textContent = "";
        line(`Done. ${got} keypairs in ${((performance.now() - t0) / 1000).toFixed(2)}s (${fmtN(last ? last.avg : 0)} keys/sec average).`, "done");
    } catch (err) {
        if (timer) clearInterval(timer);
        progress.textContent = "";
        line("error: " + (err.message || err), "r");
    }
    running = false;
    startBtn.disabled = !navigator.gpu; stopBtn.disabled = true;
});

stopBtn.addEventListener("click", () => { grinder?.stop(); });

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
