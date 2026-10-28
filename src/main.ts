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
  
  const requiredLimits: Record<string, number> = {
    maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    // Large scenes can allocate buffers above the WebGPU default 256 MiB limit
    maxBufferSize: adapter.limits.maxBufferSize,
  };
  const requiredFeatures: GPUFeatureName[] = [];
  if (adapter.features.has('shader-f16')) {
    requiredFeatures.push('shader-f16');
  }
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits,
  });
  device.addEventListener('uncapturederror', (event) => {
    const error_event = event as GPUUncapturedErrorEvent;
    console.error(`WebGPU validation error: ${error_event.error.message}`);
  });

  const canvas = document.querySelector<HTMLCanvasElement>('#webgpu-canvas');
  assert(canvas !== null);
  const context = canvas.getContext('webgpu') as GPUCanvasContext;
  
  init(canvas, context, device);
  
})();
