@group(0) @binding(0) var image: texture_2d<f32>;
@vertex fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let p=vec2<f32>(f32((index<<1u)&2u),f32(index&2u));
    return vec4<f32>(p*2.-1.,0.,1.);
}
@fragment fn fs_main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(image,vec2<i32>(p.xy),0);
}
