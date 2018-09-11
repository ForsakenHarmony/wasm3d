#[derive(Copy, Clone, Debug)]
enum Types {
  I8(i8),
  U8(u8),
  I16(i16),
  U16(u16),
  F32(f32),
  U32(u32),
}

#[derive(Debug)]
enum Container<T: Copy> {
  Scalar(T),
  Vec2([T; 2]),
  Vec3([T; 3]),
  Vec4([T; 4]),
  Mat2([[T; 2]; 2]),
  Mat3([[T; 3]; 3]),
  Mat4([[T; 4]; 4]),
}

//fn parse_buffer<T>(buffers: &Vec<::gltf::buffer::Data> , accessor: &::gltf::Accessor) -> T {
//
//}
