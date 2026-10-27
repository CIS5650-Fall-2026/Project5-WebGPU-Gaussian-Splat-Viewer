import { load } from './utils/load';
import { Camera, load_camera_presets } from './camera/camera';
import get_gaussian from './renderers/gaussian-renderer';
import get_pointcloud from './renderers/point-cloud-renderer';
import { Renderer } from './renderers/renderer';

const width = 960;
const height = 640;
const rounds = 3;
const warmup_frames = 10;
const measured_frames = 30;
const batch_size = 10;

function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  return n % 2 ? sorted[n >> 1] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
}

function download(name: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function run_performance_page(
  canvas: HTMLCanvasElement,
  context: GPUCanvasContext,
  device: GPUDevice,
  adapter: GPUAdapter,
) {
  const console_element = document.querySelector<HTMLElement>('.console');
  console_element.style.cssText = 'position:static;margin:16px;font:14px system-ui;color:#111';
  document.querySelector('#title').textContent = 'Project 5 Performance Analysis';
  const controls = document.createElement('div');
  controls.innerHTML = `
    <p>Fixed 960 x 640 output, camera preset 0, scale 1.0. Keep this tab visible and other GPU apps idle.</p>
    <label>Scene PLY <input id="scene-file" type="file" accept=".ply"></label>
    <label>Camera JSON <input id="camera-file" type="file" accept=".json"></label>
    <label>Test <select id="analysis-suite"><option value="required">Required analysis</option><option value="precision">f32 / f16 comparison</option></select></label>
    <button id="run-analysis">Run analysis</button>
    <button id="save-json" disabled>Save JSON</button>
    <button id="save-csv" disabled>Save CSV</button>
    <p id="analysis-status">Select a scene and its matching camera file.</p>
    <pre id="analysis-summary" style="white-space:pre-wrap"></pre>
    <div id="analysis-images" style="display:flex;flex-wrap:wrap;gap:12px"></div>
    <details><summary>Raw results</summary><textarea id="analysis-data" rows="10" style="width:95%" readonly></textarea></details>`;
  console_element.appendChild(controls);
  canvas.width = width;
  canvas.height = height;
  canvas.style.cssText = 'width:960px;height:640px;max-width:100%;object-fit:contain;margin:16px';
  const format = 'rgba8unorm';
  context.configure({ device, format, alphaMode: 'opaque' });
  const status = document.querySelector('#analysis-status');
  const summary = document.querySelector('#analysis-summary');
  const run = document.querySelector<HTMLButtonElement>('#run-analysis');
  const json_button = document.querySelector<HTMLButtonElement>('#save-json');
  const csv_button = document.querySelector<HTMLButtonElement>('#save-csv');
  let output: any;
  let failure = '';
  device.addEventListener('uncapturederror', (event: GPUUncapturedErrorEvent) => {
    failure = event.error.message;
    status.textContent = 'GPU error: ' + failure;
  });
  device.lost.then(info => { failure = info.message || 'GPU device lost'; });
  const check = () => {
    if (failure) throw new Error(failure);
    if (document.hidden) throw new Error('Tab became hidden. Keep it visible and run again.');
  };
  json_button.onclick = () => download(output.scene + '-performance.json', JSON.stringify(output, null, 2), 'application/json');
  csv_button.onclick = () => {
    const columns = ['condition', 'precision', 'round', 'sample', 'input_count', 'visible_count', 'workgroup_size', 'culling', 'gpu_ms', 'preprocess_ms', 'render_ms'];
    const rows = output.results.flatMap(r => r.samples.map(s => [r.condition, r.precision, s.round, s.sample, r.input_count, r.visible_count, r.workgroup_size, r.culling, s.gpu_ms, s.preprocess_ms ?? '', s.render_ms]));
    download(output.scene + '-performance.csv', [columns, ...rows].map(row => row.join(',')).join('\n'), 'text/csv');
  };

  run.onclick = async () => {
    const scene = document.querySelector<HTMLInputElement>('#scene-file').files[0];
    const camera_file = document.querySelector<HTMLInputElement>('#camera-file').files[0];
    if (!scene || !camera_file) {
      status.textContent = 'Select both files first.';
      return;
    }
    if (!device.features.has('timestamp-query')) {
      status.textContent = 'GPU timestamps are unavailable in this browser. Try a browser with timestamp-query support.';
      return;
    }
    const suite = document.querySelector<HTMLSelectElement>('#analysis-suite').value;
    if (suite === 'precision' && !device.features.has('shader-f16')) {
      status.textContent = 'Half-precision shaders are unavailable in this browser. The f32 viewer remains available.';
      return;
    }
    run.disabled = true;
    json_button.disabled = true;
    csv_button.disabled = true;
    failure = '';
    const resources: { destroy: () => void }[] = [];
    try {
      check();
      status.textContent = 'Loading scene...';
      const pc = await load(scene as unknown as string, device);
      resources.push(pc.gaussian_3d_buffer, pc.sh_buffer);
      const presets = await load_camera_presets(await camera_file.text());
      const camera = new Camera(canvas, device);
      resources.push(camera.uniform_buffer);
      camera.set_preset(presets[0]);
      const gaussian = get_gaussian(pc, device, format, camera.uniform_buffer);
      const pointcloud = get_pointcloud(pc, device, format, camera.uniform_buffer);
      const target = device.createTexture({ size: [width, height], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      resources.push(target);
      const view = target.createView();
      const queries = device.createQuerySet({ type: 'timestamp', count: batch_size * 4 });
      resources.push(queries);
      // Each frame resolves at a 256-byte-aligned offset.
      const resolve = device.createBuffer({ size: batch_size * 256, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      const readback = device.createBuffer({ size: batch_size * 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const count_readback = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      resources.push(resolve, readback, count_readback);
      const conditions = [
        { precision: 'f32' as 'f32' | 'f16', name: 'gaussian_wg256', size: 256, culling: true, count: pc.num_points, renderer: gaussian },
        { precision: 'f32' as 'f32' | 'f16', name: 'gaussian_wg64', size: 64, culling: true, count: pc.num_points, renderer: gaussian },
        { precision: 'f32' as 'f32' | 'f16', name: 'gaussian_wg128', size: 128, culling: true, count: pc.num_points, renderer: gaussian },
        { precision: 'f32' as 'f32' | 'f16', name: 'gaussian_no_culling', size: 256, culling: false, count: pc.num_points, renderer: gaussian },
        { precision: 'f32' as 'f32' | 'f16', name: 'gaussian_count25', size: 256, culling: true, count: Math.floor(pc.num_points / 4), renderer: gaussian },
        { precision: 'f32' as 'f32' | 'f16', name: 'gaussian_count50', size: 256, culling: true, count: Math.floor(pc.num_points / 2), renderer: gaussian },
        { precision: 'f32' as 'f32' | 'f16', name: 'pointcloud', size: 0, culling: false, count: pc.num_points, renderer: pointcloud },
      ];
      if (suite === 'precision') {
        conditions.splice(1);
        conditions[0].name = 'gaussian_f32';
        conditions.push({ name: 'gaussian_f16', precision: 'f16', size: 256, culling: true, count: pc.num_points, renderer: gaussian });
      }
      const info = adapter.info;
      output = {
        suite, shader_f16: device.features.has('shader-f16'),
        scene: scene.name, timestamp: new Date().toISOString(), user_agent: navigator.userAgent,
        adapter: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description },
        width, height, camera_preset: 0, gaussian_scale: 1, sh_degree: pc.sh_deg,
        timing: 'GPU timestamps from preprocess start to render end; pointcloud render pass only. Excludes loading, compilation, frame reset, readback, presentation and CPU submission.',
        sampling: 'Evenly spaced indices floor(i * full_count / count). Same source buffers remain allocated.',
        culling: 'Off disables NDC frustum rejection; behind-camera and invalid-covariance guards remain enabled.',
        rounds, warmup_frames, measured_frames, batch_size,
        results: conditions.map(c => ({ condition: c.name, precision: c.precision, input_count: c.count, workgroup_size: c.size, culling: c.culling, visible_count: 0, samples: [] })),
      };
      const draw = async (renderer: Renderer, frames: number) => {
        for (let start = 0; start < frames; start += batch_size) {
          const encoder = device.createCommandEncoder();
          for (let i = start; i < Math.min(frames, start + batch_size); i++) renderer.frame(encoder, view);
          device.queue.submit([encoder.finish()]);
          await device.queue.onSubmittedWorkDone();
          check();
        }
      };
      const capture = async (name: string) => {
        const buffer = device.createBuffer({ size: width * height * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const encoder = device.createCommandEncoder();
        encoder.copyTextureToBuffer({ texture: target }, { buffer, bytesPerRow: width * 4 }, [width, height]);
        device.queue.submit([encoder.finish()]);
        await buffer.mapAsync(GPUMapMode.READ);
        const pixels = new Uint8ClampedArray(buffer.getMappedRange().slice(0));
        buffer.unmap();
        buffer.destroy();
        const image_canvas = document.createElement('canvas');
        image_canvas.width = width;
        image_canvas.height = height;
        image_canvas.getContext('2d').putImageData(new ImageData(pixels, width, height), 0, 0);
        const link = document.createElement('a');
        link.href = image_canvas.toDataURL();
        link.download = scene.name + '-' + name + '.png';
        const image = document.createElement('img');
        image.src = link.href;
        image.alt = name;
        image.style.cssText = 'width:360px;background:black';
        link.append(image, document.createTextNode(name + ' (download)'));
        document.querySelector('#analysis-images').appendChild(link);
        return pixels;
      };
      let reference: Uint8ClampedArray;
      document.querySelector('#analysis-images').replaceChildren();
      for (let round = 0; round < rounds; round++) {
        for (let step = 0; step < conditions.length; step++) {
          const index = (step + round * 3) % conditions.length;
          const c = conditions[index];
          const result = output.results[index];
          status.textContent = `Round ${round + 1}/${rounds}: ${c.name}`;
          check();
          gaussian.set_options(c.size || 256, c.culling, c.count, c.precision);
          await draw(c.renderer, warmup_frames);
          for (let start = 0; start < measured_frames; start += batch_size) {
            const encoder = device.createCommandEncoder();
            const is_gaussian = c.renderer === gaussian;
            // Resolve immediately so each frame may reuse query indices 0..3.
            for (let i = 0; i < batch_size; i++) {
              c.renderer.frame(encoder, view, queries);
              encoder.resolveQuerySet(queries, 0, is_gaussian ? 4 : 2, resolve, i * 256);
            }
            encoder.copyBufferToBuffer(resolve, 0, readback, 0, batch_size * 256);
            device.queue.submit([encoder.finish()]);
            await readback.mapAsync(GPUMapMode.READ);
            const times = new BigUint64Array(readback.getMappedRange());
            for (let i = 0; i < batch_size; i++) {
              const offset = i * 32;
              const elapsed = (a: number, b: number) => Number(times[offset + b] - times[offset + a]) / 1e6;
              result.samples.push({ round, sample: start + i, gpu_ms: elapsed(0, is_gaussian ? 3 : 1), preprocess_ms: is_gaussian ? elapsed(0, 1) : null, render_ms: elapsed(is_gaussian ? 2 : 0, is_gaussian ? 3 : 1) });
            }
            readback.unmap();
            check();
          }
          if (c.renderer === gaussian) {
            const encoder = device.createCommandEncoder();
            encoder.copyBufferToBuffer(gaussian.count_buffer, 0, count_readback, 0, 4);
            device.queue.submit([encoder.finish()]);
            await count_readback.mapAsync(GPUMapMode.READ);
            const visible = new Uint32Array(count_readback.getMappedRange())[0];
            count_readback.unmap();
            if (round && visible !== result.visible_count) throw new Error('Visible count changed across identical runs');
            result.visible_count = visible;
          } else result.visible_count = null;
          if (round === 0) {
            const pixels = await capture(c.name);
            if (index === 0) reference = pixels;
            if (reference && c.count === pc.num_points && c.renderer === gaussian) {
              let sum = 0;
              let max = 0;
              let squared = 0;
              for (let i = 0; i < pixels.length; i++) if (i % 4 !== 3) {
                const delta = Math.abs(pixels[i] - reference[i]);
                sum += delta;
                squared += delta * delta;
                max = Math.max(max, delta);
              }
              result.image_mae_255 = sum / (width * height * 3);
              result.image_max_error_255 = max;
              const mse = squared / (width * height * 3);
              result.image_psnr_db = mse === 0 ? null : 10 * Math.log10(255 * 255 / mse);
            }
          }
          const values = result.samples.map(s => s.gpu_ms);
          result.median_gpu_ms = median(values);
          result.median_preprocess_ms = c.renderer === gaussian ? median(result.samples.map(s => s.preprocess_ms)) : null;
          summary.textContent = output.results.filter(r => r.samples.length).map(r => `${r.condition}: ${r.median_gpu_ms.toFixed(3)} ms GPU, ${r.median_preprocess_ms?.toFixed(3) ?? "n/a"} ms preprocess, ${r.samples.length} samples`).join('\n');
          await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
        }
      }
      gaussian.set_options(256, true);
      const encoder = device.createCommandEncoder();
      gaussian.frame(encoder, context.getCurrentTexture().createView());
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      check();
      output.validation = 'No uncaptured GPU errors; repeat counts consistent.';
      document.querySelector<HTMLTextAreaElement>('#analysis-data').value = JSON.stringify(output);
      json_button.disabled = false;
      csv_button.disabled = false;
      status.textContent = 'Analysis complete. Save JSON/CSV and comparison images. Reload to test another scene.';
    } catch (error) {
      status.textContent = 'Analysis failed: ' + String(error);
      console.error(error);
    } finally {
      resources.forEach(resource => resource.destroy());
    }
  };
}
