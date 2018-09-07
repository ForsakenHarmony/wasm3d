use std::rc::Rc;
use cgmath::Matrix4;

use super::webgl::*;

fn create_shader(
  gl: &WebGL2RenderingContext,
  kind: ShaderKind,
  source: &'static str,
) -> WebGLShader {
  let shader = gl.create_shader(kind);
  gl.shader_source(&shader, source);
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
  pub(crate) u_matrix: MatrixUniform,
}

impl ShaderProgram {
  pub fn new(
    gl: Rc<WebGL2RenderingContext>,
    shader_config: &ShaderConfig,
  ) -> Self {
    let vert_shader = create_shader(&gl, ShaderKind::Vertex, shader_config.vert_code);
    let frag_shader = create_shader(&gl, ShaderKind::Fragment, shader_config.frag_code);
    let program = create_program(&gl, &vert_shader, &frag_shader);

    let location = gl.get_uniform_location(&program, "u_matrix").unwrap();
    let u_matrix = MatrixUniform::new(Rc::clone(&gl), location);

    ShaderProgram { program, gl, u_matrix }
  }

  pub fn create_vertex_array(&self) -> WebGLVertexArrayObject {
    let vao = self.gl.create_vertex_array();
    self.gl.bind_vertex_array(&vao);
    vao
  }

  pub fn create_buffer(
    &self,
    size: AttributeSize,
    attribute: &'static str,
    data_type: DataType,
  ) -> VBO {
    let loc = self
        .gl
        .get_attrib_location(&self.program, attribute)
        .unwrap();
    let buffer = self.gl.create_buffer();
    self.gl.bind_buffer(BufferKind::Array, &buffer);
    self.gl.enable_vertex_attrib_array(loc);
    self
        .gl
        .vertex_attrib_pointer(loc, size, data_type, false, 0, 0);
    self.gl.unbind_buffer(BufferKind::Array);
    VBO::new(Rc::clone(&self.gl), buffer)
  }

  pub fn create_uniform<T: Uniform>(&self, name: &'static str) -> T {
    let location = self.gl.get_uniform_location(&self.program, name).unwrap();
    T::new(Rc::clone(&self.gl), location)
  }

  pub fn use_program(&self) {
    self.gl.use_program(&self.program);
  }
}


pub struct VBO {
  buffer: WebGLBuffer,
  gl: Rc<WebGL2RenderingContext>,
}

impl VBO {
  pub fn new(gl: Rc<WebGL2RenderingContext>, buffer: WebGLBuffer) -> Self {
    VBO { buffer, gl }
  }

  pub fn set_data(&self, data: &[f32]) {
    self.gl.bind_buffer(BufferKind::Array, &self.buffer);
    self
        .gl
        .buffer_data_f32(BufferKind::Array, data, DrawMode::Static);
  }

  pub fn set_data_bytes(&self, data: &[u8]) {
    self.gl.bind_buffer(BufferKind::Array, &self.buffer);
    self
        .gl
        .buffer_data_u8(BufferKind::Array, data, DrawMode::Static);
  }
}

pub struct MatrixUniform {
  uniform: WebGLUniformLocation,
  gl: Rc<WebGL2RenderingContext>,
}

pub trait Uniform {
  type Repr;
  fn new(gl: Rc<WebGL2RenderingContext>, loc: WebGLUniformLocation) -> Self;
  fn set(&self, val: Self::Repr);
}

impl Uniform for MatrixUniform {
  type Repr = Matrix4<f32>;

  fn new(gl: Rc<WebGL2RenderingContext>, uniform: WebGLUniformLocation) -> Self {
    MatrixUniform { uniform, gl }
  }

  fn set(&self, val: Self::Repr) {
    self.gl.uniform_matrix_4fv(&self.uniform, val.as_ref());
  }
}

