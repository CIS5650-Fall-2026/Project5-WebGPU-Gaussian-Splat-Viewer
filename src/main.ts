import './style.css'
import init from './renderers/renderer';
import { assert } from './utils/util';

(async () => {
  
  if (navigator.gpu === undefined) {
    const h = document.querySelector('#title') as HTMLElement;
    h.innerText = 'WebGPU is not supported in this browser.';
    return;
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });
  if (adapter === null) {
    const h = document.querySelector('#title') as HTMLElement;
    h.innerText = 'No adapter is available for WebGPU.';
    return;
  }
  
  const benchmark = new URLSearchParams(location.search).has('benchmark');
  const requiredFeatures: GPUFeatureName[] = [];
  if (benchmark && adapter.features.has('timestamp-query')) {
    requiredFeatures.push('timestamp-query');
  }
  if (adapter.features.has('shader-f16')) {
    requiredFeatures.push('shader-f16');
  }
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: { 
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxStorageBufferBindingSize: adapter.limits. maxStorageBufferBindingSize
     },
  });

  device.addEventListener('uncapturederror', (event: GPUUncapturedErrorEvent) => {
    console.error('WebGPU error:', event.error.message);
  });

  device.lost.then((info) => {
    console.error('GPU device lost:', info.reason, info.message);
  });

  const canvas = document.querySelector<HTMLCanvasElement>('#webgpu-canvas');
  assert(canvas !== null);
  const context = canvas.getContext('webgpu') as GPUCanvasContext;
  
  if (benchmark) {
    const { run_performance_page } = await import('./performance');
    run_performance_page(canvas, context, device, adapter);
  } else {
    init(canvas, context, device);
  }
  
})();