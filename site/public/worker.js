/* resenh in-browser engine: Pyodide runs the real `resenh` Python package, ONNX Runtime Web runs
 * the network (WebGPU when available, WebAssembly otherwise). Everything stays on the device. */

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/";
const ORT = new URL("ort/", self.location.href).href;  // self-hosted (same origin) so threads work

let py = null;
let web = null;
const sessions = {};
let backend = "wasm";

const post = (type, data = {}) => self.postMessage({ type, ...data });
const progress = (pct, label) => post("progress", { pct, label });

async function init(cfg) {
  progress(3, "Loading the Python runtime");
  importScripts(PYODIDE + "pyodide.js");
  importScripts(ORT + "ort.webgpu.min.js");
  ort.env.wasm.wasmPaths = ORT;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

  py = await loadPyodide({ indexURL: PYODIDE });
  progress(30, "Loading NumPy, OpenCV and Pillow");
  await py.loadPackage(["numpy", "opencv-python", "pillow"]);
  progress(70, "Installing resenh");
  // the wheel is pure Python: unpack it straight into site-packages
  const whl = await (await fetch(new URL(cfg.wheel, cfg.base).href)).arrayBuffer();
  py.unpackArchive(whl, "wheel");
  web = py.pyimport("resenh.web");
  progress(80, "Loading the neural network");
  const s = await getSession("resenh-web-hybrid", cfg);
  progress(92, "Warming up");
  // one tiny run compiles the GPU shaders now, so the first real photo is fast
  const z = (c) => new ort.Tensor("float32", new Float32Array(c * 64 * 64), [1, c, 64, 64]);
  await s.run({ x: z(3), priors: z(7) });
  progress(100, "Ready");
  post("ready", { backend, version: py.runPython("import resenh; resenh.__version__") });
}

async function getSession(name, cfg) {
  if (sessions[name]) return sessions[name];
  const url = new URL(`models/${name}.onnx?v=3`, cfg.base).href;
  const buf = new Uint8Array(await (await fetch(url)).arrayBuffer());
  const tries = [];
  if (self.navigator.gpu && !cfg.forceWasm) tries.push("webgpu");
  tries.push("wasm");
  for (const ep of tries) {
    try {
      sessions[name] = await ort.InferenceSession.create(buf, { executionProviders: [ep] });
      backend = ep;
      return sessions[name];
    } catch (e) {
      console.warn(`resenh: ${ep} unavailable`, e);
    }
  }
  throw new Error("Could not start the neural network on this device.");
}

async function runNet(req, cfg) {
  const s = await getSession(req.model, cfg);
  const feeds = { x: new ort.Tensor("float32", new Float32Array(req.x.buffer.slice(req.x.byteOffset, req.x.byteOffset + req.x.byteLength)), [1, 3, req.h, req.w]) };
  if (req.priors) {
    const p = req.priors;
    feeds.priors = new ort.Tensor("float32", new Float32Array(p.buffer.slice(p.byteOffset, p.byteOffset + p.byteLength)), [1, req.prior_channels, req.h, req.w]);
  }
  let out = (await s.run(feeds)).y;
  let data = out.data;
  if (backend === "webgpu" && !Number.isFinite(data[0])) {
    // rare WebGPU numerical failure: redo on WebAssembly
    delete sessions[req.model];
    cfg.forceWasm = true;
    return runNet(req, cfg);
  }
  return data;
}

const toJs = (p) => {
  const v = p.toJs({ dict_converter: Object.fromEntries });
  p.destroy();
  return v;
};

async function run(msg, cfg) {
  const t0 = performance.now();
  progress(10, "Analysing the photo");
  let res = toJs(web.start(msg.bytes, msg.mode, msg.preset, msg.side));
  let netMs = 0;
  if (res.need_network) {
    progress(40, "Running the neural network");
    const t = performance.now();
    const y = await runNet(res, cfg);
    netMs = performance.now() - t;
    progress(80, "Fusing and checking quality");
    res = toJs(web.finish(new Uint8Array(y.buffer, y.byteOffset, y.byteLength)));
  }
  post("result", {
    image: res.image, input: res.input, panels: res.panels, report: res.report,
    ms: performance.now() - t0, netMs, backend,
  });
}

let cfg = null;
self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "init") { cfg = m.cfg; await init(cfg); }
    else if (m.type === "run") await run(m, cfg);
  } catch (err) {
    console.error(err);
    post("error", { message: String(err && err.message ? err.message : err).split("\n").slice(-3).join(" ") });
  }
};
