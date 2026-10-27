// GPU profiling helper: timestamp queries around passes plus a few u32 counters copied from GPU buffers.
// Everything is read back asynchronously; while a readback is in flight new frames are simply not recorded.

export class GpuStats {
  readonly timing_supported: boolean;
  // smoothed results, updated whenever a readback finishes
  readonly durations_ms: number[];
  readonly counters: number[];

  private readonly query_set?: GPUQuerySet;
  private readonly resolve_buffer?: GPUBuffer;
  private readonly readback_buffer: GPUBuffer;
  private readonly counter_offset: number;
  private pending = false;
  private recorded = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly num_timestamps: number,
    // durations are measured between consecutive pairs of timestamp indices
    private readonly intervals: [number, number][],
    num_counters: number,
    private readonly smoothing = 0.9,
  ) {
    this.timing_supported = device.features.has('timestamp-query');
    this.durations_ms = intervals.map(() => 0);
    this.counters = new Array(num_counters).fill(0);
    this.counter_offset = num_timestamps * 8;

    if (this.timing_supported) {
      this.query_set = device.createQuerySet({ type: 'timestamp', count: num_timestamps });
      this.resolve_buffer = device.createBuffer({
        label: 'timestamp resolve',
        size: this.counter_offset,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
    }
    this.readback_buffer = device.createBuffer({
      label: 'stats readback',
      size: this.counter_offset + Math.max(1, num_counters) * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  // timestampWrites for a pass, or undefined when profiling is unsupported or a readback is pending
  writes(begin?: number, end?: number): GPUComputePassTimestampWrites | undefined {
    if (!this.query_set || this.pending) return undefined;
    return {
      querySet: this.query_set,
      beginningOfPassWriteIndex: begin,
      endOfPassWriteIndex: end,
    };
  }

  copy_counter(encoder: GPUCommandEncoder, index: number, src: GPUBuffer, src_offset = 0) {
    if (this.pending) return;
    encoder.copyBufferToBuffer(src, src_offset, this.readback_buffer, this.counter_offset + index * 4, 4);
  }

  // call once per frame after all passes were recorded
  resolve(encoder: GPUCommandEncoder) {
    if (this.pending) return;
    if (this.query_set && this.resolve_buffer) {
      encoder.resolveQuerySet(this.query_set, 0, this.num_timestamps, this.resolve_buffer, 0);
      encoder.copyBufferToBuffer(this.resolve_buffer, 0, this.readback_buffer, 0, this.counter_offset);
    }
    this.recorded = true;
  }

  // call after the command buffer was submitted
  after_submit() {
    if (!this.recorded || this.pending) return;
    this.recorded = false;
    this.pending = true;
    this.readback_buffer.mapAsync(GPUMapMode.READ).then(() => {
      const data = this.readback_buffer.getMappedRange();
      if (this.timing_supported) {
        const ts = new BigInt64Array(data, 0, this.num_timestamps);
        this.intervals.forEach(([a, b], i) => {
          const ms = Number(ts[b] - ts[a]) / 1e6;
          // passes skipped by the driver can produce garbage, ignore those samples
          if (ms >= 0 && ms < 1000) {
            this.durations_ms[i] = this.smoothing * this.durations_ms[i] + (1 - this.smoothing) * ms;
          }
        });
      }
      const c = new Uint32Array(data, this.counter_offset, this.counters.length);
      c.forEach((v, i) => { this.counters[i] = v; });
      this.readback_buffer.unmap();
      this.pending = false;
    }).catch(() => { this.pending = false; });
  }
}
