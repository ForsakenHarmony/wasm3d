use cgmath::{
  perspective, Deg, EuclideanSpace, Matrix, Matrix4, Point3, Rad, SquareMatrix, Transform, Vector3,
};
use std::collections::HashMap;
use std::rc::Rc;
use std::any::TypeId;
use std::marker::PhantomData;

use super::webgl::*;
use super::shader::{ShaderConfig, ShaderProgram, Uniform, UniformType};
use super::mesh::{VertexFormat, Mesh};

pub struct Camera {
  proj: Matrix4<f32>,
  view: Matrix4<f32>,
  pos: Point3<f32>,
  view_proj: Matrix4<f32>,
}

impl Camera {
  pub fn perspective(fov: Deg<f32>, aspect: f32, near: f32, far: f32, pos: Point3<f32>) -> Self {
    let proj = perspective(fov, aspect, near, far);
    let view = Matrix4::look_at(pos, Point3::origin(), Vector3::unit_y());
    let view_proj = proj * view;

    Camera {
      proj,
      pos,
      view,
      view_proj,
    }
  }

  pub fn combined(&self) -> Matrix4<f32> {
    self.view_proj
  }

  pub fn look_at(&mut self, target: Point3<f32>) {
    self.view = Matrix4::look_at(self.pos, target, Vector3::unit_y());
  }

  pub fn set_pos(&mut self, pos: Point3<f32>) {
    self.pos = pos;
  }

  pub fn update(&mut self) {
    self.view_proj = self.proj * self.view;
  }
}

#[derive(Copy, Clone, Debug)]
pub struct MeshRef(usize, TypeId);

pub struct Renderer {
  pub(crate) gl: Rc<WebGL2RenderingContext>,
  projection: Matrix4<f32>,
  size: (u32, u32),
  shaders: HashMap<TypeId, ShaderProgram>,
  shader_config: ShaderConfig,
  meshes: Vec<Mesh<Box<VertexFormat>>>,
  queue: Vec<(MeshRef, Matrix4<f32>)>,
  texture_index: usize,
}

impl Renderer {
  pub fn new(
    gl: Rc<WebGL2RenderingContext>,
    size: (u32, u32),
    shader_config: ShaderConfig,
  ) -> Self {
    Renderer {
      gl,
      projection: Matrix4::one(),
      size,
      shaders: HashMap::new(),
      shader_config,
      meshes: Vec::new(),
      queue: Vec::new(),
      texture_index: 0,
    }
  }

  pub fn aspect(&self) -> f32 {
    self.size.0 as f32 / self.size.1 as f32
  }

  pub fn set_projection(&mut self, projection: Matrix4<f32>) {
    self.projection = projection;
  }

  pub fn create_texture(&self, pixels: &[u8], width: u16, height: u16) -> WebGLTexture {
    let texture = self.gl.create_texture();
    self.gl.bind_texture(&texture);
    self.gl.tex_parameteri(
      TextureParameter::TextureMinFilter,
      TextureMinFilter::Nearest as i32,
    );
    self.gl.tex_parameteri(
      TextureParameter::TextureMagFilter,
      TextureMagFilter::Nearest as i32,
    );
    self.gl.tex_image2d(
      TextureBindPoint::Texture2d,
      0,
      PixelFormat::Rgba,
      width,
      height,
      PixelFormat::Rgba,
      DataType::U8,
      pixels,
    );
    // gl.generate_mipmap(TextureKind::Texture2d);
    texture
  }

  pub fn create_mesh<V>(&mut self, vertices: Box<V>, indices: Option<Vec<u16>>) -> MeshRef
    where V: VertexFormat + Sized + 'static,
          ::std::boxed::Box<V>: VertexFormat
  {
    // get the shader program for the vertex type, or create one
    let type_id = TypeId::of::<V>();
    let mesh = Mesh::new(&self, vertices as Box<VertexFormat>, indices);
    let program = self.shaders.entry(type_id).or_insert(ShaderProgram::new(Rc::clone(&self.gl), &self.shader_config, mesh.vertices.flags()));
    self.meshes.push(mesh);
    MeshRef(self.meshes.len() - 1, type_id)
//    mesh
  }

  pub fn clear(&self, r: f32, g: f32, b: f32, a: f32) {
    self.gl.clear_color(r, g, b, a);
    self.gl.clear(BufferBit::Color);
    self.gl.clear(BufferBit::Depth);
  }

  pub(crate) fn start(&self) {
//    self.gl.enable(Flag::CullFace);
    self.gl.enable(Flag::DepthTest);

    let fb = self.gl.create_framebuffer();

    let depth_texture = self.gl.create_texture();


//    self.gl.bind_framebuffer(Buffers::Framebuffer, &fb);

//    let depth_texture = self.gl.create_texture();
//    self.gl.bind_texture(&depth_texture);
////    self.gl.tex_storage_2d();
////    self.gl.tex_image2d(TextureBindPoint::Texture2d, 0, PixelFormat::DepthComponent, 1024, 1024, PixelFormat::DepthComponent, DataType::Float, &[]);
//    self.gl.tex_storage_2d(TextureKind::Texture2d, 1, Buffers::DepthComponent16, 1024, 1024);
  }

  pub(crate) fn exec(&mut self) {
    self.queue.sort_unstable_by_key(|(mesh, _)| mesh.1);

    if self.queue.len() > 0 {
      let first = self.queue[0].0;
      let mut active = None;

      for (MeshRef(id, shader), transform) in self.queue.iter() {
        let mut active_shader = self.shaders.get_mut(shader).expect("There should be a program for a mesh that was previously created with it");

        match active {
            Some(active) if active == *shader => {},
            _ => {
              active = Some(*shader);
              active_shader.use_program();
            },
        }

        let mesh: &Mesh<_> = self.meshes.get(*id).unwrap();

        active_shader.uniform("u_matrix", self.projection * transform);

        mesh.vao.bind(|| {
          self.gl.draw_elements(Primitives::Triangles, mesh.indices.len(), DataType::U16, 0);
        });
      }

      self.queue.clear();
    }
  }

  pub fn render_mesh(&mut self, mesh: MeshRef, transform: Matrix4<f32>) {
    self.queue.push((mesh, transform));
  }

//  pub fn render_mesh_<V: VertexFormat>(&mut self, mesh: &Mesh<V>, transform: Matrix4<f32>) where V: 'static {
//    let typeid = TypeId::of::<V>();
//    let program = self.shaders.get_mut(&TypeId::of::<V>()).expect("There should be a program for a mesh that was previously created with it");
//
//    if let Some(active) = self.active_shader {
//      if active != typeid {
//        program.use_program();
//      }
//    } else {
//      program.use_program();
//    }
//    program.u_matrix.set(self.projection * transform);
//
//    mesh.vao.bind(|| {
//      self.gl.draw_elements(Primitives::Triangles, mesh.indices.len(), DataType::U16, 0);
//    });
//  }

  pub fn create_vertex_array(&self) -> VAO {
    VAO::new(Rc::clone(&self.gl))
  }

  pub fn create_vertex_buffer<T: VBOType>(
    &self,
    data_type: DataType,
    item_size: u32,
    data: &[T],
  ) -> VBO {
    VBO::new(Rc::clone(&self.gl), data_type, item_size, data, false)
  }

  pub fn create_index_buffer<T: VBOType>(
    &self,
    data_type: DataType,
    item_size: u32,
    data: &[T],
  ) -> VBO {
    VBO::new(Rc::clone(&self.gl), data_type, item_size, data, true)
  }
}


pub struct VAO {
  gl: Rc<WebGL2RenderingContext>,
  vao: WebGLVertexArrayObject,
  num_elements: u32,
  indexed: bool,
  index_type: Option<DataType>,
  instanced: bool,
  num_instances: u32,
}

impl VAO {
  pub fn new(gl: Rc<WebGL2RenderingContext>) -> Self {
    let vao = gl.create_vertex_array();

    VAO {
      gl,
      vao,
      num_elements: 0,
      indexed: false,
      index_type: None,
      instanced: false,
      num_instances: 0,
    }
  }

  pub fn vertex_attribute_buffer(&mut self, attr_index: u32, vertex_buffer: &VBO) -> &mut Self {
    self.attribute_buffer(attr_index, vertex_buffer, false, false, false);

    self
  }

  pub fn instance_attribute_buffer(&mut self, attr_index: u32, vertex_buffer: &VBO) -> &mut Self {
    self.attribute_buffer(attr_index, vertex_buffer, true, false, false);

    self
  }

  pub fn vertex_integer_attribute_buffer(&mut self, attr_index: u32, vertex_buffer: &VBO) -> &mut Self {
    self.attribute_buffer(attr_index, vertex_buffer, false, true, false);

    self
  }

  pub fn instance_integer_attribute_buffer(&mut self, attr_index: u32, vertex_buffer: &VBO) -> &mut Self {
    self.attribute_buffer(attr_index, vertex_buffer, true, true, false);

    self
  }

  pub fn vertex_normalized_attribute_buffer(&mut self, attr_index: u32, vertex_buffer: &VBO) -> &mut Self {
    self.attribute_buffer(attr_index, vertex_buffer, false, false, true);

    self
  }

  pub fn instance_normalized_attribute_buffer(&mut self, attr_index: u32, vertex_buffer: &VBO) -> &mut Self {
    self.attribute_buffer(attr_index, vertex_buffer, true, false, true);

    self
  }

  pub fn bind<F>(&self, closure: F)
    where
        F: FnOnce() -> () {
    self.gl.bind_vertex_array(&self.vao);
    closure();
    self.gl.unbind_vertex_array();
  }

  pub fn index_buffer(&mut self, vertex_buffer: &VBO) -> &mut Self {
    self.gl.bind_vertex_array(&self.vao);
    self.gl.bind_buffer(vertex_buffer.binding, &vertex_buffer.buffer);

    self.num_elements = vertex_buffer.num_items * 3;
    self.index_type = Some(vertex_buffer.ty);
    self.indexed = true;

    self.gl.unbind_vertex_array();
    self.gl.unbind_buffer(vertex_buffer.binding);

    self
  }

  pub fn attribute_buffer(&mut self, attr_index: u32, vertex_buffer: &VBO, instanced: bool, integer: bool, normalized: bool) -> &mut Self {
    self.gl.bind_vertex_array(&self.vao);
    self.gl.bind_buffer(vertex_buffer.binding, &vertex_buffer.buffer);

    let type_size = match vertex_buffer.ty {
      DataType::I8 => 1,
      DataType::U8 => 1,
      DataType::I16 => 2,
      DataType::U16 => 2,
      DataType::I32 => 4,
      DataType::U32 => 4,
      DataType::Float => 4,
    };

    let num_columns = vertex_buffer.num_columns;

    for i in 0..num_columns {
      if integer {
        self.gl.vertex_attrib_i_pointer(
          attr_index + i,
          vertex_buffer.item_size,
          vertex_buffer.ty,
          num_columns * vertex_buffer.item_size * type_size,
          i * vertex_buffer.item_size * type_size,
        );
      } else {
        self.gl.vertex_attrib_pointer(
          attr_index + i,
          vertex_buffer.item_size,
          vertex_buffer.ty,
          normalized,
          num_columns * vertex_buffer.item_size * type_size,
          i * vertex_buffer.item_size * type_size,
        );
      }

      if instanced {
        self.gl.vertex_attrib_divisor(attr_index + i, 1);
      }

      self.gl.enable_vertex_attrib_array(attr_index + i);
    }

    self.instanced = self.instanced || instanced;

    if instanced {
      self.num_instances = vertex_buffer.num_items;
    } else {
      self.num_elements = if self.num_elements != 0 { self.num_elements } else { vertex_buffer.num_items }
    }

    self.gl.unbind_vertex_array();
    self.gl.unbind_buffer(vertex_buffer.binding);

    self
  }
}

pub enum BufferData {
  I8(Vec<i8>),
  U8(Vec<u8>),
  I16(Vec<i16>),
  U16(Vec<u16>),
  I32(Vec<i32>),
  U32(Vec<u32>),
  Float(Vec<f32>),
}

pub trait VBOType {
  fn set_data(buffer: &VBO, data: &[Self]) where Self: Sized;
}

impl VBOType for f32 {
  fn set_data(buffer: &VBO, data: &[Self]) {
    buffer.gl.buffer_data_f32(buffer.binding, data, DrawMode::Static);
  }
}

impl VBOType for u8 {
  fn set_data(buffer: &VBO, data: &[Self]) {
    buffer.gl.buffer_data_u8(buffer.binding, data, DrawMode::Static);
  }
}

impl VBOType for u16 {
  fn set_data(buffer: &VBO, data: &[Self]) {
    buffer.gl.buffer_data_u16(buffer.binding, data, DrawMode::Static);
  }
}

pub struct VBO {
  gl: Rc<WebGL2RenderingContext>,
  buffer: WebGLBuffer,
  ty: DataType,
  item_size: u32,
  num_items: u32,
  num_columns: u32,
  // usage
  index_array: bool,
  binding: BufferKind,
}

impl VBO {
  pub fn new<T: VBOType>(gl: Rc<WebGL2RenderingContext>, ty: DataType, item_size: u32, data: &[T], index_array: bool) -> Self {
    let buffer = gl.create_buffer();

    let binding = if index_array { BufferKind::ElementArray } else { BufferKind::Array };

    let buffer = VBO {
      gl,
      buffer,
      ty,
      item_size,
      num_items: 0, // TODO
      num_columns: 1, // TODO
      index_array,
      binding,
    };

    buffer.set_data(data);

    buffer
  }

  fn set_data<T: VBOType>(&self, data: &[T]) {
    self.bind(|| {
      T::set_data(&self, data);
    });
  }

  fn bind<F>(&self, closure: F)
    where
        F: FnOnce() -> () {
    self.gl.bind_buffer(self.binding, &self.buffer);
    closure();
    self.gl.unbind_buffer(self.binding);
  }
}
