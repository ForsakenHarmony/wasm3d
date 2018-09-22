use std::ops::Deref;

use super::renderer::{VBO, VAO, Renderer};
use super::webgl::*;

bitflags! {
  pub struct VertexFlags: u16 {
    const Pos = 0b00000001;
    const Tex = 0b00000010;
    const Nor = 0b00000100;
    const Col = 0b00001000;
    const Tan = 0b00010000;
    const PosTex = Self::Pos.bits | Self::Tex.bits;
    const PosCol = Self::Pos.bits | Self::Col.bits;
    const PosTexNor = Self::Pos.bits | Self::Tex.bits | Self::Nor.bits;
    const PosTexNorTan = Self::Pos.bits | Self::Tex.bits | Self::Nor.bits | Self::Tan.bits;
  }
}

impl VertexFlags {
  pub fn to_defs(&self) -> String {
    let mut defs = Vec::new();
    if self.contains(VertexFlags::Tex) {
      defs.push("#define TEXTURE")
    }
    if self.contains(VertexFlags::Nor) {
      defs.push("#define NORMALS")
    }
    if self.contains(VertexFlags::Col) {
      defs.push("#define COLOR")
    }
    if self.contains(VertexFlags::Tan) {
      defs.push("#define TANGENTS")
    }
    defs.join("\n")
  }
}

pub trait VertexFormat {
  fn flags(&self) -> VertexFlags { VertexFlags::empty() }
  fn create_buffers(&self, renderer: &Renderer, indices: &[u16]) -> (VAO, Vec<VBO>, VBO) {
    unreachable!();
  }
  fn vertex_count(&self) -> usize {
    unreachable!();
  }
}

#[derive(VertexFormat)]
#[vertex(flags = "PosTex")]
pub struct VertexPosTex {
  #[vertex(loc = 0, size = 3)]
  pub pos: Vec<f32>,
  #[vertex(loc = 1, size = 2)]
  pub tex: Vec<f32>,
}

#[derive(VertexFormat)]
#[vertex(flags = "PosCol")]
pub struct VertexPosCol {
  #[vertex(loc = 0, size = 3)]
  pub pos: Vec<f32>,
  #[vertex(loc = 3, size = 4)]
  pub col: Vec<u8>,
}

#[derive(VertexFormat)]
#[vertex(flags = "PosTexNor")]
pub struct VertexPosTexNor {
  #[vertex(loc = 0, size = 3)]
  pub pos: Vec<f32>,
  #[vertex(loc = 1, size = 2)]
  pub tex: Vec<f32>,
  #[vertex(loc = 2, size = 3)]
  pub nor: Vec<f32>,
}

struct VertexUnused;

impl VertexFormat for VertexUnused {}

impl VertexFormat for Box<VertexFormat> {
  fn flags(&self) -> VertexFlags { self.deref().flags() }
  fn create_buffers(&self, renderer: &Renderer, indices: &[u16]) -> (VAO, Vec<VBO>, VBO) {
    self.deref().create_buffers(renderer, indices)
  }
  fn vertex_count(&self) -> usize {
    self.deref().vertex_count()
  }
}

impl<T: VertexFormat> VertexFormat for Box<T> {
  fn flags(&self) -> VertexFlags { self.deref().flags() }
  fn create_buffers(&self, renderer: &Renderer, indices: &[u16]) -> (VAO, Vec<VBO>, VBO) {
    self.deref().create_buffers(renderer, indices)
  }
  fn vertex_count(&self) -> usize {
    self.deref().vertex_count()
  }
}

pub struct Mesh<V: VertexFormat + Sized> {
  pub vertices: V,
  pub indices: Vec<u16>,
  pub(crate) vao: VAO,
  pub(crate) buffers: Vec<VBO>,
  pub(crate) index_buffer: VBO,
}

impl<V: VertexFormat + Sized> Mesh<V> {
  pub fn new(renderer: &Renderer, vertices: V, indices: Option<Vec<u16>>) -> Self {
    let indices = indices.unwrap_or((0u16..vertices.vertex_count() as u16).collect());

    let vao = renderer.create_vertex_array();

    let (vao, buffers, index_buffer) = vertices.create_buffers(&renderer, indices.as_slice());

    Mesh {
      vertices,
      indices,
      vao,
      buffers,
      index_buffer,
    }
  }
}
