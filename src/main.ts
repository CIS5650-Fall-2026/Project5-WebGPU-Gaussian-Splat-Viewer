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
  
  const device = await adapter.requestDevice({
    requiredLimits: { 
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxStorageBufferBindingSize: adapter.limits. maxStorageBufferBindingSize, //max size of the regoin exposed to a shader through one storage-buffer bniding
      maxBufferSize: adapter.limits.maxBufferSize // max size of one buffer allocated with createBuffer. default is 256 MiB, 
    },
  });

  const canvas = document.querySelector<HTMLCanvasElement>('#webgpu-canvas');
  assert(canvas !== null);
  const context = canvas.getContext('webgpu') as GPUCanvasContext;
  
  init(canvas, context, device);
  
})();