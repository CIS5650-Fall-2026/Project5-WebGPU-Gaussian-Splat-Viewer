import { build } from 'esbuild';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { create, globals } from '../node_modules/gpu-check/node_modules/webgpu/index.js';
import { Float16Array } from '@petamoriken/float16';

Object.assign(globalThis, globals);
// The loader's logging and FileReader adapters do not affect GPU calculations.
globalThis.document = { querySelector: () => ({ appendChild() {} }), createElement: () => ({}) };
globalThis.FileReader = class {
  async readAsArrayBuffer(blob) {
    try { this.onload({ target: { result: await blob.arrayBuffer() } }); }
    catch (error) { this.onerror(error); }
  }
};
await mkdir('node_modules/gpu-check', { recursive: true });
await build({ stdin: { contents: `
export {default as tiles} from './src/renderers/tile-renderer.ts';
export {default as renderer} from './src/renderers/gaussian-renderer.ts';
export {get_sorter} from './src/sort/sort.ts';
export {load} from './src/utils/load.ts';
export {Camera,load_camera_presets} from './src/camera/camera.ts';
`, resolveDir: process.cwd() },
  bundle: true, platform: 'node', format: 'esm', loader: { '.wgsl': 'text' },
  outfile: 'node_modules/gpu-check/renderers.mjs' });
const api = await import('../node_modules/gpu-check/renderers.mjs');
const gpu = create(['backend=d3d12']);
const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
if (!adapter) throw Error('No Dawn GPU adapter');
const report = { runtime: 'Dawn Node WebGPU / D3D12', elapsed_note: 'Validation wall time includes readback, CPU comparison, and lazy setup; not a steady-state benchmark', date: new Date().toISOString(), synthetic: [], scenes: [], errors: [] };
async function deviceFor(half = true) {
  const current = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  const device = await current.requestDevice({ requiredFeatures: half ? ['shader-f16'] : [],
    requiredLimits: { maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
  device.pushErrorScope('validation');
  return device;
}
async function finish(device) {
  const error = await device.popErrorScope();
  if (error) throw Error(error.message);
  if (report.errors.length) throw Error(report.errors.join('\n'));
  device.destroy();
}
async function pixels(device, width, height, draw) {
  const texture = device.createTexture({ size: [width, height], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const stride = Math.ceil(width * 4 / 256) * 256;
  const buffer = device.createBuffer({ size: stride * height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder();
  draw(encoder, texture.createView());
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: stride }, [width, height]);
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const source = new Uint8Array(buffer.getMappedRange());
  const result = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) result.set(source.subarray(y * stride, y * stride + width * 4), y * width * 4);
  buffer.unmap(); buffer.destroy(); texture.destroy();
  return result;
}
function compare(actual, expected) {
  let total = 0, max = 0, lit = 0;
  for (let i = 0; i < actual.length; i++) if (i % 4 !== 3) {
    const delta = Math.abs(actual[i] - expected[i]);
    total += delta; max = Math.max(max, delta); lit += actual[i] !== 0;
  }
  return { mean: total / (actual.length / 4 * 3), max, lit };
}
const bits = x => new Uint32Array(new Float32Array([x]).buffer)[0];
async function synthetic(count, half, overflow = false) {
  const device = await deviceFor(half);
  const sorter = api.get_sorter(Math.max(1, count), device);
  const stride = half ? 48 : 64;
  const data = new ArrayBuffer(Math.max(1, count) * stride);
  const splats = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const ids = new Uint32Array(Math.max(1, count)), depths = new Uint32Array(Math.max(1, count));
  const radius = overflow ? 256 : 32;
  const input = [];
  for (let i = 0; i < count; i++) {
    // Shuffled depths, ties, and low opacity make missing tail entries observable.
    const depth = i === count - 1 ? .5 : 1 + (i * 103 % 251);
    const rgba = [(i % 7) / 7, (i % 11) / 11, (i % 13) / 13, i === count - 1 ? .8 : count > 1000 ? .004 : .2];
    const tail = [0.01, 0, 0.01, 0, ...rgba];
    new Float32Array(data, i * stride, 8).set([0, 0, .5, 1, 1, 1, radius, 0]);
    const packed = half ? new Float16Array(data, i * stride + 32, 8) : new Float32Array(data, i * stride + 32, 8);
    packed.set(tail);
    input.push({ id: i, depth, conic: Number(packed[0]), color: Array.from(packed.slice(4)) });
    ids[i] = i; depths[i] = ~bits(depth);
  }
  device.queue.writeBuffer(splats, 0, data);
  const tiles = api.tiles(device, 'rgba8unorm', splats, sorter, count, half);
  for (const [width, height] of overflow ? [[129, 129]] : [[33, 19], [47, 31]]) {
    tiles.resize(width, height);
    device.queue.writeBuffer(sorter.ping_pong[0].sort_indices_buffer, 0, ids);
    device.queue.writeBuffer(sorter.ping_pong[0].sort_depths_buffer, 0, depths);
    device.queue.writeBuffer(sorter.sort_info_buffer, 0, new Uint32Array([count]));
    const actual = await pixels(device, width, height, (enc, view) => { tiles.bin(enc); sorter.sort(enc); tiles.frame(enc, view); });
    const expected = new Uint8Array(actual.length);
    input.sort((a, b) => a.depth - b.depth || b.id - a.id);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      let trans = 1; const color = [0, 0, 0];
      const dx = x + .5 - width / 2, dy = y + .5 - height / 2;
      for (const s of input) {
        if (Math.abs(dx) > radius || Math.abs(dy) > radius) continue;
        const alpha = Math.min(.99, s.color[3] * Math.exp(-.5 * s.conic * (dx * dx + dy * dy)));
        if (alpha < 1 / 255) continue;
        for (let c = 0; c < 3; c++) color[c] += trans * alpha * s.color[c];
        trans *= 1 - alpha;
        if (trans < .0001) break;
      }
      expected.set([...color.map(c => Math.round(c * 255)), 255], (y * width + x) * 4);
    }
    const diff = compare(actual, expected);
    // Tied depths use the original global sort in the pool-overflow path.
    if (diff.max > 1) throw Error(JSON.stringify({ count, half, width, overflow, ...diff }));
    report.synthetic.push({ count, half, width, height, overflow, ...diff, passed: true });
  }
  tiles.destroy();
  await finish(device);
  console.log('synthetic passed', count, half, overflow);
}
async function scene(full = false, halfFeature = true) {
  const width = full ? 1280 : 640, height = full ? 720 : 360;
  const device = await deviceFor(halfFeature);
  const camera = new api.Camera({ width, height }, device);
  camera.set_preset((await api.load_camera_presets(await readFile('assets/scenes/bicycle-cameras.json')))[0]);
  const pc = await api.load(await readFile(full ? 'assets/scenes/bicycle.ply' : 'assets/scenes/bicycle-smoke.ply'), device);
  const renderer = api.renderer(pc, device, 'rgba8unorm', camera.uniform_buffer);
  renderer.set_viewport(width, height);
  const render = () => pixels(device, width, height, (enc, view) => renderer.frame(enc, view));
  const baseline = await render();
  for (const tile of [false, true]) for (const half of halfFeature ? [false, true, false] : [false]) for (const size of full ? [256] : [64, 128, 256]) {
    renderer.set_tile_sorting(tile); renderer.set_half_precision(half); renderer.set_workgroup_size(size);
    const start = performance.now();
    const actual = await render();
    const diff = compare(actual, baseline);
    if (!diff.lit || diff.mean > 1 || diff.max > 100) throw Error(JSON.stringify({ full, tile, half, size, ...diff }));
    report.scenes.push({ full, halfFeature, points: pc.num_points, width, height, tile, half, size, elapsed_ms: performance.now() - start, ...diff });
    console.log('scene passed', full, tile, half, size, diff);
    if (full && tile && half) await writeFile('node_modules/gpu-check/full-scene.rgba', actual);
  }
  renderer.set_scale(0);
  const empty = await render();
  if (empty.some((v, i) => i % 4 !== 3 && v !== 0)) throw Error('zero scale failed');
  renderer.set_scale(1);
  const recovered = compare(await render(), baseline);
  if (!recovered.lit || recovered.mean > 1) throw Error('recovery failed');
  if (!halfFeature) {
    let rejected = false;
    try { renderer.set_half_precision(true); } catch { rejected = true; }
    if (!rejected) throw Error('unsupported f16 accepted');
  }
  await finish(device);
}
try {
  const mode = process.argv[2] || 'synthetic';
  if (mode === 'synthetic') {
    for (const half of [false, true]) for (const count of [0, 2, 1024, 1025, 4097]) await synthetic(count, half);
    await synthetic(64, false, true);
  } else if (mode === 'sample') await scene();
  else if (mode === 'full') await scene(true);
  else if (mode === 'no-f16') await scene(false, false);
  else throw Error('Unknown validation mode');
  await writeFile(`performance/gpu-validation-${mode}.json`, JSON.stringify(report, null, 2) + '\n');
  console.log('PASS', mode);
  process.exit(0);
} catch (error) {
  console.error(error);
  process.exit(1);
}
