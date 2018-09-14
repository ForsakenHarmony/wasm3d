use std::rc::Rc;
use std::marker::Sized;
use std::collections::HashMap;

use cgmath::Matrix4;

use super::webgl::*;
use super::mesh::VertexFormat;
use super::Result;

fn create_shader(
  gl: &WebGL2RenderingContext,
  kind: ShaderKind,
  source: String,
) -> WebGLShader {
  let shader = gl.create_shader(kind);
  gl.shader_source(&shader, &source);
  gl.compile_shader(&shader);
  shader
}

fn create_program(
  gl: &WebGL2RenderingContext,
  vertex: &WebGLShader,
  fragment: &WebGLShader,
) -> WebGLProgram {
  let program = gl.create_program();
  gl.attach_shader(&program, vertex);
  gl.attach_shader(&program, fragment);
  gl.link_program(&program);
  program
}

pub struct ShaderConfig {
  vert_code: &'static str,
  frag_code: &'static str,
}

impl Default for ShaderConfig {
  fn default() -> Self {
    ShaderConfig {
      vert_code: include_str!("./shaders/vert.glsl"),
      frag_code: include_str!("./shaders/frag.glsl"),
    }
  }
}

pub struct ShaderProgram {
  pub(crate) program: WebGLProgram,
  pub(crate) gl: Rc<WebGL2RenderingContext>,
  pub(crate) u_matrix: Uniform,
  pub(crate) uniforms: HashMap<String, Uniform>,
}

impl ShaderProgram {
  pub fn new<V: VertexFormat>(
    gl: Rc<WebGL2RenderingContext>,
    shader_config: &ShaderConfig,
  ) -> Self {
    let definitions = V::flags().to_defs();

    let vert_code = shader_config.vert_code.replace("#{{defs}}", &definitions);
    let frag_code = shader_config.frag_code.replace("#{{defs}}", &definitions);

    let vert_shader = create_shader(&gl, ShaderKind::Vertex, vert_code);
    let frag_shader = create_shader(&gl, ShaderKind::Fragment, frag_code);
    let program = create_program(&gl, &vert_shader, &frag_shader);

    let location = gl.get_uniform_location(&program, "u_matrix").unwrap();
    let u_matrix = Uniform::new(Rc::clone(&gl), location);

    let mut uniforms = HashMap::new();

    let num_uniforms = gl.get_program_parameter(&program, ShaderParameter::ActiveUniforms);

    for i in 0..num_uniforms as u32 {
      let info = gl.get_active_uniform(&program, i);
      let handle = gl.get_uniform_location(&program, &info.name()).unwrap();

      let wrapper = Uniform::new(Rc::clone(&gl), handle);

      uniforms.insert(info.name().clone(), wrapper);
    }

    ShaderProgram { program, gl, u_matrix, uniforms }
  }

  pub fn create_vertex_array(&self) -> VAO {
    VAO::new(Rc::clone(&self.gl))
  }

  pub fn create_vertex_buffer<T: VBOType>(
    &self,
    data_type: DataType,
    item_size: u32,
    data: &[T]
  ) -> VBO {
    VBO::new(Rc::clone(&self.gl), data_type, item_size, data, false)
  }

  pub fn create_index_buffer<T: VBOType>(
    &self,
    data_type: DataType,
    item_size: u32,
    data: &[T]
  ) -> VBO {
    VBO::new(Rc::clone(&self.gl), data_type, item_size, data, true)
  }

  pub fn uniform<T: UniformType>(&mut self, name: &'static str, value: T) {
    match self.uniforms.get_mut(name) {
      Some(ref uniform) => uniform.set(value),
      None => {},
    };
  }

  pub fn use_program(&self) {
    self.gl.use_program(&self.program);
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

pub struct Uniform {
  gl: Rc<WebGL2RenderingContext>,
  uniform: WebGLUniformLocation,
}

impl Uniform {
  fn new(gl: Rc<WebGL2RenderingContext>, uniform: WebGLUniformLocation) -> Self {
    Uniform { uniform, gl }
  }

  pub fn set<T: UniformType>(&self, val: T) {
    T::set(&self, val);
  }
}

pub trait UniformType {
  fn set(uniform: &Uniform, val: Self);
}

impl UniformType for Matrix4<f32> {
  fn set(uniform: &Uniform, val: Self) {
    uniform.gl.uniform_matrix_4fv(&uniform.uniform, val.as_ref());
  }
}

//pub trait Uniform {
//  type Repr;
//  fn new(gl: Rc<WebGL2RenderingContext>, loc: WebGLUniformLocation) -> Self;
//  fn set(&self, val: Self::Repr);
//}
//
//impl Uniform for MatrixUniform {
//  type Repr = Matrix4<f32>;
//
//  fn new(gl: Rc<WebGL2RenderingContext>, uniform: WebGLUniformLocation) -> Self {
//    MatrixUniform { uniform, gl }
//  }
//
//  fn set(&self, val: Self::Repr) {
//    self.gl.uniform_matrix_4fv(&self.uniform, val.as_ref());
//  }
//}

