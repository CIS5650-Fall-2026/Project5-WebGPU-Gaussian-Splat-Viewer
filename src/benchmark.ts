import type { Renderer } from './renderers/renderer';

export async function benchmark(renderer: Renderer, device: GPUDevice,
  format: GPUTextureFormat, width: number, height: number) {
  const texture = device.createTexture({
    label: 'benchmark target',
    size: [width, height],
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  const view = texture.createView();
  const samples: number[] = [];
  try {
    await device.queue.onSubmittedWorkDone();
    // Wait for each frame so these numbers include submission and GPU completion.
    for (let i = 0; i < 150; i++) {
      const start = performance.now();
      const encoder = device.createCommandEncoder();
      renderer.frame(encoder, view);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      if (i >= 30) samples.push(performance.now() - start);
    }
    const sorted = [...samples].sort((a, b) => a - b);
    return {
      method: 'CPU wall time per submitted frame through GPU completion; no presentation',
      warmup_frames: 30,
      measured_frames: samples.length,
      mean_ms: samples.reduce((a, b) => a + b, 0) / samples.length,
      median_ms: (sorted[59] + sorted[60]) / 2,
      p95_ms: sorted[Math.ceil(sorted.length * 0.95) - 1],
      samples_ms: samples,
    };
  } finally {
    texture.destroy();
  }
}
