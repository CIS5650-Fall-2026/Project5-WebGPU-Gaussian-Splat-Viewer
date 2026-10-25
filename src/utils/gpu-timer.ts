
export class GPUTimer {
  readonly querySet: GPUQuerySet;
  readonly resolveBuffer: GPUBuffer;
  readonly readBuffer: GPUBuffer;

  constructor(device: GPUDevice, queryCount = 2) {
    this.querySet = device.createQuerySet({
      type: 'timestamp',
      count: queryCount,
    });

    const byteSize = queryCount * BigUint64Array.BYTES_PER_ELEMENT;

    this.resolveBuffer = device.createBuffer({
      label: 'GPU timestamp resolve buffer',
      size: byteSize,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });

    this.readBuffer = device.createBuffer({
      label: 'GPU timestamp readback buffer',
      size: byteSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  resolve(encoder: GPUCommandEncoder, queryCount = 2) {
    encoder.resolveQuerySet(
      this.querySet,
      0,
      queryCount,
      this.resolveBuffer,
      0
    );

    encoder.copyBufferToBuffer(
      this.resolveBuffer,
      0,
      this.readBuffer,
      0,
      queryCount * BigUint64Array.BYTES_PER_ELEMENT
    );
  }

  async read(): Promise<number[]> {
    await this.readBuffer.mapAsync(GPUMapMode.READ);

    const values = new BigUint64Array(
      this.readBuffer.getMappedRange().slice(0)
    );

    this.readBuffer.unmap();

    return Array.from(values, value => Number(value));
  }

  destroy() {
    this.querySet.destroy();
    this.resolveBuffer.destroy();
    this.readBuffer.destroy();
  }
}
