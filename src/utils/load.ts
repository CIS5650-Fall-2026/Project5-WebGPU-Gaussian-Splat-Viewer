import { Float16Array } from '@petamoriken/float16';
import { log, time, timeLog } from './simple-console';
import { decodeHeader, readRawVertex ,nShCoeffs} from './plyreader';

const c_size_float = 2;   // byte size of f16

const c_size_3d_gaussian =
  3 * c_size_float  // x y z (position)
  + c_size_float    // opacity
  + 4 * c_size_float  // rotation
  + 4 * c_size_float  //scale
;

export type PointCloud = Awaited<ReturnType<typeof load>>;

export async function load(file: string, device: GPUDevice) {
  const blob = new Blob([file]);
  const arrayBuffer = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = function(event) {
      resolve(event.target.result);  // Resolve the promise with the ArrayBuffer
    };

    reader.onerror = reject;  // Reject the promise in case of an error
    reader.readAsArrayBuffer(blob);
  });

  const [vertexCount, propertyTypes, vertexData] = decodeHeader(arrayBuffer as ArrayBuffer);
  // figure out the SH degree from the number of coefficients
  var nRestCoeffs = 0;
  for (const propertyName in propertyTypes) {
      if (propertyName.startsWith('f_rest_')) {
          nRestCoeffs += 1;
      }
  }
  const nCoeffsPerColor = nRestCoeffs / 3;
  const sh_deg = Math.sqrt(nCoeffsPerColor + 1) - 1;
  console.log("Sh Deg: " + sh_deg);
  const num_coefs = nShCoeffs(sh_deg);
  const max_num_coefs = 16;

  const c_size_sh_coef = 
    3 * max_num_coefs * c_size_float // 3 channels (RGB) x 16 coefs
  ;

  // figure out the order in which spherical harmonics should be read
  const shFeatureOrder = [];
  for (let rgb = 0; rgb < 3; ++rgb) {
      shFeatureOrder.push(`f_dc_${rgb}`);
  }
  for (let i = 0; i < nCoeffsPerColor; ++i) {
      for (let rgb = 0; rgb < 3; ++rgb) {
          shFeatureOrder.push(`f_rest_${rgb * nCoeffsPerColor + i}`);
      }
  }

  const num_points = vertexCount;

  log(`num points: ${num_points}`);
  log(`processing loaded attributes...`);
  time();

  // xyz (position), opacity, cov (from rot and scale)
  const gaussian_3d_buffer = device.createBuffer({
    label: 'ply input 3d gaussians data buffer',
    size: num_points * c_size_3d_gaussian,  // buffer size multiple of 4?
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const gaussian = new Float16Array(gaussian_3d_buffer.getMappedRange());

  // Spherical harmonic function coeffs
  const sh_buffer = device.createBuffer({
    label: 'ply input 3d gaussians data buffer',
    size: num_points * c_size_sh_coef,  // buffer size multiple of 4?
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const sh = new Float16Array(sh_buffer.getMappedRange());

  var readOffset = 0;
  for (let i = 0; i < num_points; i++) {
    const [newReadOffset, rawVertex] = readRawVertex(readOffset, vertexData, propertyTypes);
    readOffset = newReadOffset;

    const o = i * (c_size_3d_gaussian / c_size_float);
    const output_offset = i * max_num_coefs * 3;
    
    for (let order = 0; order < num_coefs; ++order) {
        const order_offset = order * 3;
        for (let j = 0; j < 3; ++j) {
            const coeffName = shFeatureOrder[order * 3 + j];
            sh[output_offset +order_offset+j]=rawVertex[coeffName];
        }
    }

    gaussian[o + 0] = rawVertex.x;
    gaussian[o + 1] = rawVertex.y;
    gaussian[o + 2] = rawVertex.z;
    gaussian[o + 3] = rawVertex.opacity;
    gaussian[o + 4] = rawVertex.rot_0;
    gaussian[o + 5] = rawVertex.rot_1;
    gaussian[o + 6] = rawVertex.rot_2;
    gaussian[o + 7] = rawVertex.rot_3;
    gaussian[o + 8] = rawVertex.scale_0;
    gaussian[o + 9] = rawVertex.scale_1;
    gaussian[o + 10] = rawVertex.scale_2;
  }

  gaussian_3d_buffer.unmap(); 
  sh_buffer.unmap();

  timeLog();
  console.log("return result!");
  return {
    num_points: num_points,
    sh_deg: sh_deg,
    gaussian_3d_buffer,
    sh_buffer,
  };
}
