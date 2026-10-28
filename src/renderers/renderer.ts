import { load } from '../utils/load';
import { Pane } from 'tweakpane';
import * as TweakpaneFileImportPlugin from 'tweakpane-plugin-file-import';
import { default as get_renderer_gaussian, GaussianRenderer } from './gaussian-renderer';
import { default as get_renderer_pointcloud } from './point-cloud-renderer';
import { Camera, load_camera_presets} from '../camera/camera';
import { CameraControl } from '../camera/camera-control';
import { time, timeReturn } from '../utils/simple-console';

export interface Renderer {
  frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => void,
  camera_buffer: GPUBuffer,
}

export default async function init(
  canvas: HTMLCanvasElement,
  context: GPUCanvasContext,
  device: GPUDevice
) {
  let ply_file_loaded = false; 
  let cam_file_loaded = false; 
  let renderers: { pointcloud?: Renderer, gaussian?: Renderer } = {};
  let gaussian_renderer: GaussianRenderer | undefined; 
  let pointcloud_renderer: Renderer | undefined; 
  let renderer: Renderer | undefined; 
  let cameras;
  
  const camera = new Camera(canvas, device);
  const control = new CameraControl(camera);

  const observer = new ResizeObserver(() => {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    camera.on_update_canvas();
  });
  observer.observe(canvas);
  
  const presentation_format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: presentation_format,
    alphaMode: 'opaque',
  });
  

  // Tweakpane: easily adding tweak control for parameters.
  const params = {
    fps: 0.0,
    gaussian_multiplier: 1,
    sh_degree: 0,
    frustum_culling: true,
    gaussian_stride: 1,
    active_gaussians: 0,
    renderer: 'pointcloud',
    ply_file: '',
    cam_file: '',
  };

  const pane = new Pane({
    title: 'Config',
    expanded: true,
  });
  pane.registerPlugin(TweakpaneFileImportPlugin);
  let sh_degree_input: ReturnType<typeof pane.addInput> | undefined;
  const update_sh_degree_control = (available_degree: number): void => {
    sh_degree_input?.dispose();
  
    const max_degree = Math.max(
      0,
      Math.min(3, Math.floor(available_degree)),
    );
  
    params.sh_degree = Math.max(
      0,
      Math.min(params.sh_degree, max_degree),
    );
  
    const options: Record<string, number> = {};
  
    for (let degree = 0; degree <= max_degree; degree++) {
      options[degree] = degree;
    }
  
    sh_degree_input = pane.addInput(params, 'sh_degree', {
      label: 'SH degree',
      options,
    }).on('change', (e) => {
      gaussian_renderer?.set_sh_degree(e.value);
    });
  };
  {
    pane.addMonitor(params, 'fps', {
      readonly:true
    });
  }
  {
    pane.addInput(params, 'renderer', {
      options: {
        pointcloud: 'pointcloud',
        gaussian: 'gaussian',
      }
    }).on('change', (e) => {
      renderer = renderers[e.value];
    });
  }
  {
    pane.addInput(params, 'ply_file', {
      view: 'file-input',
      lineCount: 3,
      filetypes: ['.ply'],
      invalidFiletypeMessage: "We can't accept those filetypes!"
    })
    .on('change', async (file) => {
      const uploadedFile = file.value;
      if (uploadedFile) {
        const pc = await load(uploadedFile, device);
        pointcloud_renderer = get_renderer_pointcloud(pc, device, presentation_format, camera.uniform_buffer);
        gaussian_renderer = get_renderer_gaussian(pc, device, presentation_format, camera.uniform_buffer);
        gaussian_renderer.set_gaussian_scaling(params.gaussian_multiplier);
        update_sh_degree_control(pc.sh_deg);
        gaussian_renderer.set_sh_degree(params.sh_degree);
        gaussian_renderer.set_frustum_culling(params.frustum_culling);
        params.active_gaussians = gaussian_renderer.set_gaussian_stride(params.gaussian_stride);
        renderers = {
          pointcloud: pointcloud_renderer,
          gaussian: gaussian_renderer,
        };
        renderer = renderers[params.renderer];
        ply_file_loaded = true;
      }else{
        ply_file_loaded = false;
      }
    });
  }
  {
    pane.addInput(params, 'cam_file', {
      view: 'file-input',
      lineCount: 3,
      filetypes: ['.json'],
      invalidFiletypeMessage: "We can't accept those filetypes!"
    })
    .on('change', async (file) => {
      const uploadedFile = file.value;
      if (uploadedFile) {
        cameras=await load_camera_presets(file.value);
        camera.set_preset(cameras[0]);
        cam_file_loaded = true;
      }else{
        cam_file_loaded = false;
      }
    });
  }
  {
    pane.addInput(
      params,
      'gaussian_multiplier',
      { min: 0, max: 1.5 },
    ).on('change', (e) => {
      gaussian_renderer?.set_gaussian_scaling(e.value);
    });
  }
  {
    pane.addInput(params, 'frustum_culling', {
      label: 'Frustum culling',
    }).on('change', (e) => {
      gaussian_renderer?.set_frustum_culling(e.value);
    });
  }
  {
    pane.addInput(params, 'gaussian_stride', {
      label: 'Gaussian fraction',
      options: {
        '100%': 1,
        '50%': 2,
        '25%': 4,
      },
    }).on('change', (e) => {
      if (gaussian_renderer) {
        params.active_gaussians =
          gaussian_renderer.set_gaussian_stride(e.value);
      }
    });

    pane.addMonitor(params, 'active_gaussians', {
      label: 'Input Gaussians',
      format: (value) => value.toFixed(0),
    });
  }

  document.addEventListener('keydown', (event) => {
    switch(event.key) {
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
        const i = parseInt(event.key);
        console.log(`set to camera preset ${i}`);
        camera.set_preset(cameras[i]);
        break;
    }
  });

  function frame() {
    if (ply_file_loaded && cam_file_loaded) {
      params.fps=1.0/timeReturn()*1000.0;
      time();
      const encoder = device.createCommandEncoder();
      const texture_view = context.getCurrentTexture().createView();
      renderer.frame(encoder, texture_view);
      device.queue.submit([encoder.finish()]);
    }
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}
