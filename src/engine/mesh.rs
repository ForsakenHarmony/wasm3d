use super::shader::{VBO, ShaderProgram};
use super::webgl::*;

bitflags! {
  pub struct VertexFlags: u16 {
    const Pos = 0b00000001;
    const Tex = 0b00000010;
    const Nor = 0b00000100;
    const Col = 0b00001000;
    const Tan = 0b00010000;
    const PosTex = Self::Pos.bits | Self::Tex.bits;
    const PosTexNor = Self::Pos.bits | Self::Tex.bits | Self::Nor.bits;
    const PosTexNorTan = Self::Pos.bits | Self::Tex.bits | Self::Nor.bits | Self::Tan.bits;
  }
}

impl VertexFlags {
  pub fn to_defs(&self) -> String {
    let mut defs = Vec::new();
//    let mut defs = String::new();
    if self.contains(VertexFlags::Tex) {
      defs.push("#define TEXTURE\n")
    }
    if self.contains(VertexFlags::Nor) {
      defs.push("#define NORMALS\n")
    }
    if self.contains(VertexFlags::Col) {
      defs.push("#define COLOR\n")
    }
    defs.join("\n")
  }
}

pub trait VertexFormat {
  type Buffers;
  const FLAGS: VertexFlags;

  fn create_buffers(&self, program: &ShaderProgram) -> Self::Buffers where Self: ::std::marker::Sized;
  fn vertex_count(&self) -> usize;
}

#[derive(VertexFormat)]
pub struct VertexPosTex {
//  #[vertex(name="a_position", size="3")]
  pub pos: Vec<f32>,
//  #[vertex(name="a_texcoord", size="2")]
  pub tex: Vec<f32>,
}

pub struct VertexPosTexNor {
  pub pos: Vec<f32>,
  pub tex: Vec<f32>,
  pub nor: Vec<f32>,
}

impl VertexFormat for VertexPosTex {
  type Buffers = (VBO, VBO);
  const FLAGS: VertexFlags = VertexFlags::PosTex;

  fn create_buffers(&self, program: &ShaderProgram) -> Self::Buffers {
    let pos_buffer = program.create_buffer(AttributeSize::Three, "a_position", DataType::Float);
    let tex_buffer = program.create_buffer(AttributeSize::Two, "a_texcoord", DataType::Float);

    pos_buffer.set_data_f32(self.pos.as_slice());
    tex_buffer.set_data_f32(self.tex.as_slice());

    (pos_buffer, tex_buffer)
  }

  fn vertex_count(&self) -> usize {
    self.pos.len() / 3
  }
}

pub struct Mesh<V: VertexFormat> {
  pub vertices: V,
  pub indices: Vec<u16>,
  pub(crate) vao: WebGLVertexArrayObject,
  pub(crate) buffers: V::Buffers,
  pub(crate) index_buffer: WebGLBuffer,
}

impl<V: VertexFormat> Mesh<V> {
  pub fn new(program: &ShaderProgram, vertices: V, indices: Option<Vec<u16>>) -> Self {
    let indices = indices.unwrap_or((0u16..vertices.vertex_count() as u16).collect());

    let vao = program.create_vertex_array();
    let buffers = vertices.create_buffers(&program);

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
