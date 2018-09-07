use super::shader::{VBO, ShaderProgram};
use super::webgl::*;

bitflags! {
  pub struct VertexFlags: u16 {
    const Pos = 0b00000001;
    const Tex = 0b00000010;
    const Nor = 0b00000100;
    const PosTex = Self::Pos.bits | Self::Tex.bits;
  }
}

pub trait VertexFormat {
  type Buffers;
  const FLAGS: VertexFlags;

  fn create_buffers(program: &ShaderProgram, vertices: &Vec<Self>) -> Self::Buffers where Self: ::std::marker::Sized;
}

pub struct VertexPosTex {
  pub pos: [f32; 3],
  pub tex: [f32; 2],
}

pub struct VertexPosTexNor {
  pub pos: [f32; 3],
  pub tex: [f32; 2],
  pub nor: [f32; 3],
}

impl VertexFormat for VertexPosTex {
  type Buffers = (VBO, VBO);
  const FLAGS: VertexFlags = VertexFlags::PosTex;

  fn create_buffers(program: &ShaderProgram, vertices: &Vec<Self>) -> Self::Buffers {
    let pos_buffer = program.create_buffer(AttributeSize::Three, "a_position", DataType::Float);
    let tex_buffer = program.create_buffer(AttributeSize::Two, "a_texcoord", DataType::Float);

    let (positions, tex_coords) = vertices.iter().fold((Vec::new(), Vec::new()), |(mut pos, mut tex), val| {
      pos.extend_from_slice(&val.pos);
      tex.extend_from_slice(&val.tex);
      (pos, tex)
    });

    pos_buffer.set_data(positions.as_slice());
    tex_buffer.set_data(tex_coords.as_slice());

    (pos_buffer, tex_buffer)
  }
}

pub struct Mesh<V: VertexFormat> {
  pub vertices: Vec<V>,
  pub indices: Vec<u16>,
  pub(crate) vao: WebGLVertexArrayObject,
  pub(crate) buffers: V::Buffers,
  pub(crate) index_buffer: WebGLBuffer,
}

impl<V: VertexFormat> Mesh<V> {
  pub fn new(program: &ShaderProgram, vertices: Vec<V>, indices: Option<Vec<u16>>) -> Self {
    let indices = indices.unwrap_or((0u16..vertices.len() as u16).collect());

    let vao = program.create_vertex_array();
    let buffers = V::create_buffers(program, &vertices);

    let index_buffer = program.gl.create_buffer();
    program.gl.bind_buffer(BufferKind::ElementArray, &index_buffer);
    program.gl.buffer_data_u16(BufferKind::ElementArray, indices.as_slice(), DrawMode::Static);

    Mesh {
      vertices,
      indices,
      vao,
      buffers,
      index_buffer,
    }
  }
}
