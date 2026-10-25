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
  
  
  const requiredFeatures: GPUFeatureName[] = [];

  if (adapter.features.has('timestamp-query')) {
    requiredFeatures.push('timestamp-query');
  }

  if (adapter.features.has('shader-f16')) {
    requiredFeatures.push('shader-f16');
  }

  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });

  console.log(
    'GPU timestamp queries enabled:',
    device.features.has('timestamp-query')
  );

  console.log(
    'GPU shader-f16 enabled:',
    device.features.has('shader-f16')
  );

  const canvas = document.querySelector<HTMLCanvasElement>('#webgpu-canvas');
  assert(canvas !== null);
  const context = canvas.getContext('webgpu') as GPUCanvasContext;
  
  init(canvas, context, device);
  
})();