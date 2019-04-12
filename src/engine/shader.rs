use std::rc::Rc;
use std::marker::Sized;
use std::collections::HashMap;

use cgmath::Matrix4;

use super::webgl::*;
use super::mesh::{VertexFormat, VertexFlags};
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
      vert_code: include_str!("shaders/main_vert.glsl"),
      frag_code: include_str!("shaders/main_frag.glsl"),
    }
  }
}

pub struct ShaderProgram {
  pub(crate) program: WebGLProgram,
  pub(crate) gl: Rc<WebGL2RenderingContext>,
  pub(crate) uniforms: HashMap<String, Uniform>,
}

impl ShaderProgram {
  pub fn new(
    gl: Rc<WebGL2RenderingContext>,
    shader_config: &ShaderConfig,
    flags: VertexFlags,
  ) -> Self {
    let definitions = flags.to_defs();

    let vert_code = shader_config.vert_code.replace("#{{defs}}", &definitions);
    let frag_code = shader_config.frag_code.replace("#{{defs}}", &definitions);

    let vert_shader = create_shader(&gl, ShaderKind::Vertex, vert_code);
    let frag_shader = create_shader(&gl, ShaderKind::Fragment, frag_code);
    let program = create_program(&gl, &vert_shader, &frag_shader);

    let mut uniforms = HashMap::new();

    let num_uniforms = gl.get_program_parameter(&program, ShaderParameter::ActiveUniforms);

    for i in 0..num_uniforms as u32 {
      let info = gl.get_active_uniform(&program, i);
      let handle = gl.get_uniform_location(&program, &info.name()).unwrap();

      let wrapper = Uniform::new(Rc::clone(&gl), handle);

      uniforms.insert(info.name().clone(), wrapper);
    }

    ShaderProgram { program, gl, uniforms }
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

