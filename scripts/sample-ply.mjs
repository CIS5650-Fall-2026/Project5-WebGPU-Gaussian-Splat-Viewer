import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, basename, extname, join } from 'node:path';
import { createHash } from 'node:crypto';

// Keep every nth complete Gaussian record, including all SH coefficients.
// This samples across the source ordering rather than taking a prefix.
const [sourceArg, outputArg, ...strideArgs] = process.argv.slice(2);
if (!sourceArg || !outputArg) {
  console.error('Usage: node scripts/sample-ply.mjs <input.ply> <output-directory> [strides: 2 4]');
  process.exitCode = 1;
} else {
  const source = resolve(sourceArg);
  const output = resolve(outputArg);
  const strides = strideArgs.length ? strideArgs.map(Number) : [2, 4];
  if (strides.some(n => !Number.isSafeInteger(n) || n < 2) || new Set(strides).size !== strides.length) {
    throw new Error('Strides must be distinct integers of at least 2.');
  }
  const data = await readFile(source);
  const headerEnd = data.indexOf(Buffer.from('end_header\n'));
  if (headerEnd < 0) throw new Error('Expected a PLY header ending in end_header followed by LF.');
  const offset = headerEnd + Buffer.byteLength('end_header\n');
  const header = data.subarray(0, offset).toString('ascii');
  const lines = header.trimEnd().split(/\r?\n/);
  if (lines[0] !== 'ply' || !lines.includes('format binary_little_endian 1.0')) {
    throw new Error('Only binary little-endian PLY 1.0 files are supported.');
  }
  const elements = lines.filter(line => line.startsWith('element '));
  if (elements.length !== 1 || !/^element vertex \d+$/.test(elements[0])) {
    throw new Error('Expected a vertex-only PLY; other elements are not sampled.');
  }
  const count = Number(elements[0].split(' ')[2]);
  const scalarBytes = { char:1, int8:1, uchar:1, uint8:1, short:2, int16:2,
    ushort:2, uint16:2, int:4, int32:4, uint:4, uint32:4, float:4, float32:4, double:8, float64:8 };
  let recordBytes = 0;
  for (const line of lines.filter(line => line.startsWith('property '))) {
    const match = /^property (\w+) (\w+)$/.exec(line);
    if (!match || !scalarBytes[match[1]]) throw new Error('Unsupported property: ' + line);
    recordBytes += scalarBytes[match[1]];
  }
  if (!count || !recordBytes || data.length - offset !== count * recordBytes) {
    throw new Error('PLY body length does not match its vertex count and scalar properties.');
  }
  const stem = basename(source, extname(source));
  const manifest = { sourceName:basename(source), sourceSha256:createHash('sha256').update(data).digest('hex'),
    sourceCount:count, recordBytes, method:'Keep source record indices 0, stride, 2*stride, ...; copy full records unchanged. Ordering may be spatially correlated, so these are controlled subsets, not unbiased random samples.', subsets:[] };
  await mkdir(output, { recursive:true });
  for (const stride of strides) {
    const sampledCount = Math.ceil(count / stride);
    const sampledHeader = Buffer.from(header.replace(/^element vertex \d+$/m, 'element vertex ' + sampledCount), 'ascii');
    const sampled = Buffer.allocUnsafe(sampledHeader.length + sampledCount * recordBytes);
    sampledHeader.copy(sampled);
    let target = sampledHeader.length;
    for (let index = 0; index < count; index += stride) {
      const begin = offset + index * recordBytes;
      data.copy(sampled, target, begin, begin + recordBytes);
      target += recordBytes;
    }
    if (target !== sampled.length) throw new Error('Sampled output length mismatch.');
    const name = `${stem}.stride-${stride}.ply`;
    const destination = join(output, name);
    if (resolve(destination) === source) throw new Error('Refusing to overwrite the source.');
    // Exclusive creation avoids overwriting an existing experiment.
    await writeFile(destination, sampled, { flag:'wx' });
    manifest.subsets.push({ file:name, stride, count:sampledCount, fraction:sampledCount/count,
      bytes:sampled.length, sha256:createHash('sha256').update(sampled).digest('hex') });
    console.log(`${name}: ${sampledCount.toLocaleString()} Gaussians`);
  }
  const manifestPath = join(output, `${stem}.sampling.json`);
  await writeFile(manifestPath, JSON.stringify(manifest, null, 2)+'\n', { flag:'wx' });
  console.log('Manifest: ' + manifestPath);
}
