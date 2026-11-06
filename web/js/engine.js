// WebGPU vanity grinder engine (browser and Node with a WebGPU implementation).
import { mainShader, tableShader, combCfg } from "./kernels.js";
import * as R from "./ref.js";

const MATCH_CAP = 4096;
const REC = 12;   // u32 per match record: idx, pattern, pk[8], pad[2]

export class Grinder {
    // logNK: starting batch size; maxLogNK: the batch doubles while a batch takes < 100 ms on average
    // (browsers add per-batch latency, so fast GPUs need big batches; slow ones must keep them small).
    static async create({ gpu = globalThis.navigator?.gpu, W = 14, logNK = 20, maxLogNK = 22, mc = 64, dump = false } = {}) {
        if (!gpu) throw new Error("WebGPU is not available in this browser");
        const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
        if (!adapter) throw new Error("No WebGPU adapter found");
        const device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                maxBufferSize: adapter.limits.maxBufferSize,
                maxStorageBuffersPerShaderStage: Math.min(10, adapter.limits.maxStorageBuffersPerShaderStage),
            },
        });
        const g = new Grinder();
        g.adapter = adapter; g.device = device;
        // any validation / out-of-memory error must stop the run, never be skipped silently
        device.addEventListener("uncapturederror", (ev) => { g.gpuError = ev.error?.message || String(ev.error); });
        device.lost.then((info) => { g.gpuError = g.gpuError || `GPU device lost: ${info.message || info.reason}`; });
        g.info = adapter.info || {};
        g.cfg = combCfg(W);
        const lim = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
        const maxGroups = adapter.limits.maxComputeWorkgroupsPerDimension;   // 65535 by default
        while (maxLogNK > logNK && ((1 << maxLogNK) * 32 > lim || (1 << maxLogNK) / 128 > maxGroups)) maxLogNK--;
        g.nk = 1 << logNK; g.capNK = 1 << Math.max(logNK, maxLogNK); g.mc = mc;
        g.dumpMode = dump;
        await g._init();
        return g;
    }

    async _init() {
        const d = this.device, cfg = this.cfg;
        const t0 = performance.now();
        // ---- compile everything in parallel (the slow part on a first visit) ----
        const tmod = d.createShaderModule({ code: tableShader(cfg) });
        const mmod = d.createShaderModule({ code: mainShader(cfg) });
        await Promise.all([this._checkModule(tmod, "table"), this._checkModule(mmod, "main")]);
        const S = GPUShaderStage.COMPUTE;
        const ro = { type: "read-only-storage" }, rw = { type: "storage" };
        this.bgl = d.createBindGroupLayout({ entries: [
            { binding: 0, visibility: S, buffer: { type: "uniform" } },
            { binding: 1, visibility: S, buffer: ro }, { binding: 2, visibility: S, buffer: rw },
            { binding: 3, visibility: S, buffer: rw }, { binding: 4, visibility: S, buffer: rw },
            { binding: 5, visibility: S, buffer: rw }, { binding: 6, visibility: S, buffer: ro },
            { binding: 7, visibility: S, buffer: ro }, { binding: 8, visibility: S, buffer: rw },
            { binding: 9, visibility: S, buffer: rw }] });
        const layout = d.createPipelineLayout({ bindGroupLayouts: [this.bgl] });
        const [tpipe, genPipe, invPipe] = await Promise.all([
            d.createComputePipelineAsync({ layout: "auto", compute: { module: tmod, entryPoint: "table_main" } }),
            d.createComputePipelineAsync({ layout, compute: { module: mmod, entryPoint: "gen_main" } }),
            d.createComputePipelineAsync({ layout, compute: { module: mmod, entryPoint: "inv_main" } })]);
        this.genPipe = genPipe; this.invPipe = invPipe;
        this.compileMs = performance.now() - t0;
        // ---- table ----
        const bases = R.combBases(cfg.w, cfg.nwin);
        const bw = new Uint32Array(bases.length * 16);
        bases.forEach(([x, y], k) => { bw.set(R.bigToWords(x), k * 16); bw.set(R.bigToWords(y), k * 16 + 8); });
        const basesBuf = this._buf(bw.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, bw);
        this.table = this._buf(cfg.entries * 96, GPUBufferUsage.STORAGE);
        const tpBuf = this._buf(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        const tbg = d.createBindGroup({ layout: tpipe.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: tpBuf } }, { binding: 1, resource: { buffer: basesBuf } },
            { binding: 2, resource: { buffer: this.table } }] });
        const CH = 8192;
        for (let first = 0; first < cfg.entries; first += CH) {
            const cnt = Math.min(CH, cfg.entries - first);
            d.queue.writeBuffer(tpBuf, 0, new Uint32Array([first, cnt, 0, 0]));
            const enc = d.createCommandEncoder();
            const p = enc.beginComputePass();
            p.setPipeline(tpipe); p.setBindGroup(0, tbg); p.dispatchWorkgroups(Math.ceil(cnt / 64)); p.end();
            d.queue.submit([enc.finish()]);
            await d.queue.onSubmittedWorkDone();
        }
        basesBuf.destroy();
        this.tableMs = performance.now() - t0;

        // ---- buffers ----
        const fe = this.capNK * 32, ST = GPUBufferUsage.STORAGE;
        this.X = this._buf(fe, ST); this.Y = this._buf(fe, ST); this.Z = this._buf(fe, ST); this.Pf = this._buf(fe, ST);
        this.params = this._buf(112, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.outBytes = (4 + MATCH_CAP * REC) * 4;
        this.out = this._buf(this.outBytes, ST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        this.dump = this._buf(this.dumpMode ? this.nk * 32 : 16, ST | GPUBufferUsage.COPY_SRC);
        this.key = new Uint32Array(8);
        crypto.getRandomValues(this.key);
        this.nonce = 0;
        this.next = 0n;
        this.initMs = performance.now() - t0;
    }

    async _checkModule(m, name) {
        if (!m.getCompilationInfo) return;
        const info = await m.getCompilationInfo();
        const errs = info.messages.filter((x) => x.type === "error");
        if (errs.length) throw new Error(`${name} shader: ` + errs.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join("\n"));
    }

    _buf(size, usage, data) {
        const b = this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage, mappedAtCreation: !!data });
        if (data) { new Uint32Array(b.getMappedRange()).set(data); b.unmap(); }
        return b;
    }

    setPatterns(list, icase) {
        const pats = R.buildPatterns(list, icase);
        const pw = new Uint32Array(Math.max(1, list.length) * 90);
        list.forEach(({ prefix = "", suffix = "" }, i) => {
            const enc = (ch) => { const v = R.B58.indexOf(ch); return icase ? R.FOLD[v] : v; };
            pw[i * 90] = prefix.length; pw[i * 90 + 1] = suffix.length;
            [...prefix].forEach((c, q) => { pw[i * 90 + 2 + q] = icase ? foldChar(c) : enc(c); });
            [...suffix].forEach((c, q) => { pw[i * 90 + 46 + q] = icase ? foldChar(c) : enc(c); });
        });
        const rw = new Uint32Array(Math.max(1, pats.ranges.length) * 4);
        pats.ranges.forEach(([lo, hi], i) => {
            rw[i * 4] = Number(lo >> 32n); rw[i * 4 + 1] = Number(lo & 0xffffffffn);
            rw[i * 4 + 2] = Number(hi >> 32n); rw[i * 4 + 3] = Number(hi & 0xffffffffn);
        });
        this.pats = pats;
        this.patBuf?.destroy(); this.rangeBuf?.destroy();
        this.patBuf = this._buf(pw.byteLength, GPUBufferUsage.STORAGE, pw);
        this.rangeBuf = this._buf(rw.byteLength, GPUBufferUsage.STORAGE, rw);
        this.bg = this.device.createBindGroup({ layout: this.bgl, entries: [
            this.params, this.table, this.X, this.Y, this.Z, this.Pf, this.patBuf, this.rangeBuf, this.out, this.dump,
        ].map((buffer, binding) => ({ binding, resource: { buffer } })) });
        return pats;
    }

    _writeParams(base, nk) {
        const u = new Uint32Array(28);
        u.set(this.key, 0);
        u.set(this.pats.bitmap, 8);
        u[16] = Number(base & 0xffffffffn); u[17] = Number(base >> 32n); u[18] = this.nonce; u[19] = nk;
        u[20] = nk / this.mc; u[21] = this.mc; u[22] = this.pats.list.length; u[23] = this.pats.icase ? 1 : 0;
        u[24] = this.pats.ranges.length; u[25] = MATCH_CAP; u[26] = this.dumpMode ? 1 : 0; u[27] = 0;
        this.device.queue.writeBuffer(this.params, 0, u);
    }

    // Enqueue one batch; resolves with { base, count, records } once results are readable.
    _batch() {
        const d = this.device, base = this.next, nk = this.nk;
        this.next += BigInt(nk);
        this._writeParams(base, nk);
        d.queue.writeBuffer(this.out, 0, new Uint32Array(4));
        const staging = d.createBuffer({ size: this.outBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const enc = d.createCommandEncoder();
        let p = enc.beginComputePass();
        p.setPipeline(this.genPipe); p.setBindGroup(0, this.bg); p.dispatchWorkgroups(Math.ceil(nk / 128)); p.end();
        p = enc.beginComputePass();
        p.setPipeline(this.invPipe); p.setBindGroup(0, this.bg); p.dispatchWorkgroups(Math.ceil(nk / this.mc / 64)); p.end();
        enc.copyBufferToBuffer(this.out, 0, staging, 0, this.outBytes);
        d.queue.submit([enc.finish()]);
        return staging.mapAsync(GPUMapMode.READ).then(() => {
            const a = new Uint32Array(staging.getMappedRange().slice(0));
            staging.unmap(); staging.destroy();
            const count = Math.min(a[0], MATCH_CAP), records = [];
            for (let s = 0; s < count; s++) {
                const o = 4 + s * REC;
                records.push({ idx: a[o], pat: a[o + 1], pk: a.slice(o + 2, o + 10) });
            }
            return { base, nk, count: a[0], records };
        });
    }

    // Re-derive the seed on the CPU and recompute the key with the reference implementation.
    async verify(base, rec) {
        const index = base + BigInt(rec.idx);
        const seed = R.chacha12Seed(this.key, index, this.nonce);
        const pk = await R.pubkeyFromSeed(seed);
        const gpk = new Uint8Array(rec.pk.buffer.slice(0));
        if (!pk.every((b, i) => b === gpk[i])) return null;
        const address = R.b58encode(pk);
        if (!R.matches(address, this.pats)) return null;
        return { address, seed, pk, secret: R.keypairBase58(seed, pk), json: JSON.stringify([...seed, ...pk]) };
    }

    // Runs until stop(); onMatch(match), onStats({keys, rate}).
    async run({ onMatch, onStats, onError, inflight = 3 }) {
        this.running = true;
        let keys = 0, mismatch = 0;
        const t0 = performance.now();
        let lastT = t0, lastK = 0;
        const queue = [];
        try {
            while (this.running || queue.length) {
                while (this.running && queue.length < inflight) queue.push(this._batch());
                const r = await queue.shift();
                if (this.gpuError) throw new Error(this.gpuError);
                keys += r.nk;
                for (const rec of r.records) {
                    const m = await this.verify(r.base, rec);
                    if (m) onMatch?.(m); else mismatch++;
                }
                const now = performance.now();
                if (now - lastT > 1000 || !this.running) {
                    const avg = (keys * 1000) / (now - t0);
                    // batches finishing back-to-back at the end would show a fake spike: report the average then
                    const rate = this.running ? ((keys - lastK) * 1000) / (now - lastT) : avg;
                    // wall time per batch bounds its GPU time from above: keep batches well under the driver watchdog
                    if (this.running && this.nk < this.capNK && (this.nk / rate) * 1000 < 100) this.nk *= 2;
                    onStats?.({ keys, rate, avg, mismatch, dropped: r.count > MATCH_CAP });
                    lastT = now; lastK = keys;
                }
            }
        } catch (e) { onError?.(e); }
        this.running = false;
    }
    stop() { this.running = false; }

    async readDump() {
        const d = this.device, size = this.nk * 32;
        const st = d.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const enc = d.createCommandEncoder();
        enc.copyBufferToBuffer(this.dump, 0, st, 0, size);
        d.queue.submit([enc.finish()]);
        await st.mapAsync(GPUMapMode.READ);
        const a = new Uint32Array(st.getMappedRange().slice(0));
        st.unmap(); st.destroy();
        return a;
    }
    destroy() { this.device.destroy(); }
}

function foldChar(c) { return R.FOLD[R.B58.indexOf(c)] ?? R.FOLD[R.B58.indexOf(c.toLowerCase())] ?? R.FOLD[R.B58.indexOf(c.toUpperCase())]; }

// GPU keys vs reference, GPU matcher vs a full scan. Returns { ok, lines }.
export async function selftest({ gpu, W = 14 } = {}) {
    const lines = [];
    let ok = true;
    // RFC 8032 test 1
    const seed = Uint8Array.from("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60".match(/../g), (h) => parseInt(h, 16));
    const want = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a";
    const got = [...(await R.pubkeyFromSeed(seed))].map((b) => b.toString(16).padStart(2, "0")).join("");
    lines.push(`${got === want ? "ok  " : "FAIL"} reference vs RFC 8032`); ok &&= got === want;

    const g = await Grinder.create({ gpu, W, logNK: 16, maxLogNK: 16, mc: 32, dump: true });
    const list = [{ suffix: "Ab" }, { prefix: "Bc" }, { prefix: "9", suffix: "z" }, { prefix: "x" }];
    g.setPatterns(list, true);
    const r = await g._batch();
    const dump = await g.readDump();
    let bad = 0;
    for (let j = 0; j < 160; j++) {
        const i = j < 32 ? j : (j * 2654435761) % g.nk;
        const pk = await R.pubkeyFromSeed(R.chacha12Seed(g.key, BigInt(i), g.nonce));
        const gw = new Uint8Array(dump.buffer, i * 32, 32);
        if (!pk.every((b, q) => b === gw[q])) bad++;
    }
    lines.push(`${bad ? "FAIL" : "ok  "} GPU keys vs reference (160 checked, ${bad} bad)`); ok &&= !bad;
    const wantIdx = [];
    for (let i = 0; i < g.nk; i++) if (R.matches(R.b58encode(new Uint8Array(dump.buffer, i * 32, 32)), g.pats)) wantIdx.push(i);
    const gotIdx = r.records.map((x) => x.idx).sort((a, b) => a - b);
    const same = wantIdx.length === gotIdx.length && wantIdx.every((v, i) => v === gotIdx[i]);
    lines.push(`${same ? "ok  " : "FAIL"} GPU matcher vs full scan of ${g.nk} keys (${wantIdx.length} expected, ${gotIdx.length} found)`);
    ok &&= same;
    g.destroy();
    return { ok, lines };
}
