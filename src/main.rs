#![feature(proc_macro)]
#![feature(nll)]
#![recursion_limit = "512"]

// FIXME: remove this
#![allow(unused, unused_mut, dead_code)]

#[macro_use]
extern crate stdweb;
#[macro_use]
extern crate stdweb_derive;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate rand;
extern crate image;
extern crate cgmath;

mod engine;
mod util;

use engine::webgl::*;
use engine::app::*;
use cgmath::{Matrix4, Vector3, Point3, Rad, Deg, SquareMatrix, Transform, EuclideanSpace, Matrix, perspective};

fn create_shader(gl: &WebGL2RenderingContext, kind: ShaderKind, source: &'static str) -> WebGLShader {
  let shader = gl.create_shader(kind);
  gl.shader_source(&shader, source);
  gl.compile_shader(&shader);
  shader
}

fn create_program(gl: &WebGL2RenderingContext, vertex: &WebGLShader, fragment: &WebGLShader) -> WebGLProgram {
  let program = gl.create_program();
  gl.attach_shader(&program, vertex);
  gl.attach_shader(&program, fragment);
  gl.link_program(&program);
  program
}

fn create_texture(gl: &WebGL2RenderingContext, width: u16, height: u16, pixels: &[u8]) -> WebGLTexture {
  let texture = gl.create_texture();
  gl.bind_texture(&texture);
  gl.tex_image2d(TextureBindPoint::Texture2d, 0, width, height, PixelFormat::Rgba, DataType::U8, pixels);
  gl.generate_mipmap(TextureKind::Texture2d);
  texture
}

fn load_image(buffer: &[u8]) -> Result<(u16, u16, Vec<u8>), Box<std::error::Error>> {
  let img = image::load_from_memory(buffer)?.to_rgba();
  Ok((img.width() as u16, img.height() as u16, img.into_raw()))
}

struct MatrixUniform(WebGLUniformLocation);

trait Uniform {
  type Repr;
  fn new(loc: WebGLUniformLocation) -> Self;
  fn set(&self, gl: &WebGL2RenderingContext, val: Self::Repr);
}

impl Uniform for MatrixUniform {
  type Repr = Matrix4<f32>;

  fn new(loc: WebGLUniformLocation) -> Self {
    MatrixUniform(loc)
  }

  fn set(&self, gl: &WebGL2RenderingContext, val: Self::Repr) {
    gl.uniform_matrix_4fv(&self.0, val.as_ref());
  }
}

fn create_uniform<T: Uniform>(gl: &WebGL2RenderingContext, program: &WebGLProgram, name: &str) -> T {
  let location = gl.get_uniform_location(&program, name).unwrap();
  T::new(location)
}

fn create_and_bind_buffer(gl: &WebGL2RenderingContext, size: AttributeSize, data: &[f32], loc: u32) -> WebGLBuffer {
  let buffer = gl.create_buffer();
  gl.bind_buffer(BufferKind::Array, &buffer);
  gl.buffer_data_float(BufferKind::Array, data, DrawMode::Static);
  gl.enable_vertex_attrib_array(loc);
  gl.vertex_attrib_pointer(loc, size, DataType::Float, false, 0, 0);
  buffer
}

struct Camera {
  proj: Matrix4<f32>,
  view: Matrix4<f32>,
}

impl Camera {
  fn perspective(fov: Deg<f32>, aspect: f32, near: f32, far: f32, view: Matrix4<f32>) -> Self {
    let projection_matrix = perspective(Deg(60.0), aspect, 1.0, 2000.0,);

    Camera {
      proj: perspective(fov, aspect, near, far),
      view
    }
  }
}

struct Renderer {
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  camera: Camera,
}

impl Renderer {
  pub fn new(gl: WebGL2RenderingContext, vert_code: &'static str, frag_code: &'static str, camera: Camera) -> Self {
    let vert_shader = create_shader(&gl, ShaderKind::Vertex, vert_code);
    let frag_shader = create_shader(&gl, ShaderKind::Fragment, frag_code);
    let program = create_program(&gl, &vert_shader, &frag_shader);

    Renderer {
      gl,
      program,
      camera
    }
  }

  pub fn render(&self) {
    self.gl.use_program(&self.program);
  }
}

#[cfg(target_arch = "wasm32")]
pub fn main() -> Result<(), Box<std::error::Error>> {
  let size = (1280, 720);
  let config = AppConfig::new("Test", size);
  let app = App::new(config);

  let img = load_image(include_bytes!("../static/f-texture.png"))?;

  let gl = WebGL2RenderingContext::new(app.canvas());

//  let mut rng = rand::thread_rng();

  let vert_code = include_str!("../shaders/vert.glsl");
  let frag_code = include_str!("../shaders/frag.glsl");

  let vert_shader = create_shader(&gl, ShaderKind::Vertex, vert_code);

  let frag_shader = create_shader(&gl, ShaderKind::Fragment, frag_code);

  let program = create_program(&gl, &vert_shader, &frag_shader);

  // look up data locations
  let pos_attr_loc = gl.get_attrib_location(&program, "a_position").unwrap();
  let tex_coord_loc = gl.get_attrib_location(&program, "a_texcoord").unwrap();

  let u_matrix = create_uniform::<MatrixUniform>(&gl, &program, "u_matrix");

  // look up uniforms
//  let matrix_loc = gl.get_uniform_location(&program, "u_matrix").unwrap();

  let vao = gl.create_vertex_array();
  gl.bind_vertex_array(&vao);

  create_and_bind_buffer(&gl, AttributeSize::Three, get_geometry().as_slice(), pos_attr_loc);
  create_and_bind_buffer(&gl, AttributeSize::Two, get_texcoords(), tex_coord_loc);

//  let pos_buffer = gl.create_buffer();
//  gl.bind_buffer(BufferKind::Array, &pos_buffer);
//  gl.buffer_data_float(BufferKind::Array, get_geometry(), DrawMode::Static);
//  gl.enable_vertex_attrib_array(pos_attr_loc);
//  gl.vertex_attrib_pointer(pos_attr_loc, AttributeSize::Three, DataType::Float, false, 0, 0);
//
//  let tex_coord_buffer = gl.create_buffer();
//  gl.bind_buffer(BufferKind::Array, &tex_coord_buffer);
//  gl.buffer_data_float(BufferKind::Array, get_texcoords(), DrawMode::Static);
//  gl.enable_vertex_attrib_array(tex_coord_loc);
//  gl.vertex_attrib_pointer(tex_coord_loc, AttributeSize::Two, DataType::Float, false, 0, 0);

  let texture = create_texture(&gl, img.0,img.1, img.2.as_slice());

//  let texture = gl.create_texture();
//  gl.active_texture(TextureIndex::Texture0);
//  gl.bind_texture(&texture);
//
//  gl.tex_parameteri(TextureParameter::TextureWrapS, TextureWrap::ClampToEdge as i32);
//  gl.tex_parameteri(TextureParameter::TextureWrapT, TextureWrap::ClampToEdge as i32);
//  gl.tex_parameteri(TextureParameter::TextureMinFilter, TextureMinFilter::Nearest as i32);
//  gl.tex_parameteri(TextureParameter::TextureMagFilter, TextureMagFilter::Nearest as i32);
//
//  gl.tex_image2d(TextureBindPoint::Texture2d, 0, 240, 180, PixelFormat::Rgba, DataType::U8, img.raw_pixels().as_slice());

  let mut translation = Vector3::from((-150.0, 0.0, -660.0));
  let mut scale = Vector3::from((1.0, 1.0, 1.0));
  let mut angle = 0.0;

  fn to_rad(angle: f32) -> f32 {
    (360.0 - angle) * std::f32::consts::PI / 180.0
  }

  let mut last = 0.0;
  app.run(move |app: &mut App, t: f64| {
    let t = t as f32;
    let delta = (t - last) / 1000.0;
    last = t;

    angle = (angle + 10.0 * delta) % 360.0;
    let radian_angle = Deg(angle);

    gl.viewport(0, 0, size.0, size.1);

    gl.clear_color(0.0, 0.0, 0.0, 0.0);
    gl.clear(BufferBit::Color);
    gl.clear(BufferBit::Depth);

    gl.enable(Flag::CullFace);
    gl.enable(Flag::DepthTest);

    gl.use_program(&program);

    gl.bind_vertex_array(&vao);

    let num_fs = 5;
    let radius = 200.0;

    let aspect = size.0 as f32 / size.1 as f32;

    let f_pos = Point3::new(radius, 0.0, 0.0);

    let projection_matrix = perspective(Deg(60.0), aspect, 1.0, 2000.0,);

    let camera_matrix = Matrix4::from_angle_y(radian_angle) * Matrix4::from_translation(Vector3::new(0.0, 50.0, radius * 1.5));

    let cam_pos = camera_matrix.transform_point(Point3::origin());

    let view_matrix = Matrix4::look_at(cam_pos, f_pos, Vector3::unit_y());

    let view_projection_matrix = projection_matrix * view_matrix;

    for i in 0..num_fs {
      let angle = i as f32 * ::std::f32::consts::PI * 2.0 / num_fs as f32;

      let x = angle.cos() * radius;
      let z = angle.sin() * radius;

      let matrix = view_projection_matrix * Matrix4::from_translation(Vector3::new(x, 0.0, z));

      u_matrix.set(&gl, matrix);

      gl.draw_arrays(Primitives::Triangles, 16 * 6);
    }
  });

  Ok(())
}

fn get_geometry() -> Vec<f32> {
  let arr = [
    // left column front
    0.0,   0.0,  0.0,
    0.0, 150.0,  0.0,
    30.0,   0.0,  0.0,
    0.0, 150.0,  0.0,
    30.0, 150.0,  0.0,
    30.0,   0.0,  0.0,

    // top rung front
    30.0,   0.0,  0.0,
    30.0,  30.0,  0.0,
    100.0,   0.0,  0.0,
    30.0,  30.0,  0.0,
    100.0,  30.0,  0.0,
    100.0,   0.0,  0.0,

    // middle rung front
    30.0,  60.0,  0.0,
    30.0,  90.0,  0.0,
    67.0,  60.0,  0.0,
    30.0,  90.0,  0.0,
    67.0,  90.0,  0.0,
    67.0,  60.0,  0.0,

    // left column back
    0.0,   0.0,  30.0,
    30.0,   0.0,  30.0,
    0.0, 150.0,  30.0,
    0.0, 150.0,  30.0,
    30.0,   0.0,  30.0,
    30.0, 150.0,  30.0,

    // top rung back
    30.0,   0.0,  30.0,
    100.0,   0.0,  30.0,
    30.0,  30.0,  30.0,
    30.0,  30.0,  30.0,
    100.0,   0.0,  30.0,
    100.0,  30.0,  30.0,

    // middle rung back
    30.0,  60.0,  30.0,
    67.0,  60.0,  30.0,
    30.0,  90.0,  30.0,
    30.0,  90.0,  30.0,
    67.0,  60.0,  30.0,
    67.0,  90.0,  30.0,

    // top
    0.0,   0.0,   0.0,
    100.0,   0.0,   0.0,
    100.0,   0.0,  30.0,
    0.0,   0.0,   0.0,
    100.0,   0.0,  30.0,
    0.0,   0.0,  30.0,

    // top rung right
    100.0,   0.0,   0.0,
    100.0,  30.0,   0.0,
    100.0,  30.0,  30.0,
    100.0,   0.0,   0.0,
    100.0,  30.0,  30.0,
    100.0,   0.0,  30.0,

    // under top rung
    30.0,   30.0,   0.0,
    30.0,   30.0,  30.0,
    100.0,  30.0,  30.0,
    30.0,   30.0,   0.0,
    100.0,  30.0,  30.0,
    100.0,  30.0,   0.0,

    // between top rung and middle
    30.0,   30.0,   0.0,
    30.0,   60.0,  30.0,
    30.0,   30.0,  30.0,
    30.0,   30.0,   0.0,
    30.0,   60.0,   0.0,
    30.0,   60.0,  30.0,

    // top of middle rung
    30.0,   60.0,   0.0,
    67.0,   60.0,  30.0,
    30.0,   60.0,  30.0,
    30.0,   60.0,   0.0,
    67.0,   60.0,   0.0,
    67.0,   60.0,  30.0,

    // right of middle rung
    67.0,   60.0,   0.0,
    67.0,   90.0,  30.0,
    67.0,   60.0,  30.0,
    67.0,   60.0,   0.0,
    67.0,   90.0,   0.0,
    67.0,   90.0,  30.0,

    // bottom of middle rung.
    30.0,   90.0,   0.0,
    30.0,   90.0,  30.0,
    67.0,   90.0,  30.0,
    30.0,   90.0,   0.0,
    67.0,   90.0,  30.0,
    67.0,   90.0,   0.0,

    // right of bottom
    30.0,   90.0,   0.0,
    30.0,  150.0,  30.0,
    30.0,   90.0,  30.0,
    30.0,   90.0,   0.0,
    30.0,  150.0,   0.0,
    30.0,  150.0,  30.0,

    // bottom
    0.0,   150.0,   0.0,
    0.0,   150.0,  30.0,
    30.0,  150.0,  30.0,
    0.0,   150.0,   0.0,
    30.0,  150.0,  30.0,
    30.0,  150.0,   0.0,

    // left side
    0.0,   0.0,   0.0,
    0.0,   0.0,  30.0,
    0.0, 150.0,  30.0,
    0.0,   0.0,   0.0,
    0.0, 150.0,  30.0,
    0.0, 150.0,   0.0,
  ];

  let matrix = Matrix4::from_angle_x(Deg(180.0)) * Matrix4::from_translation(Vector3::new(-50.0, -75.0, -15.0));

  let mut vec = Vec::<f32>::new();

  for coord in arr.chunks(3) {
    let out: [f32; 3] = matrix.transform_point([coord[0], coord[1], coord[2]].into()).into();
    vec.extend_from_slice(&out);
  }

  vec
}

fn get_texcoords() -> &'static [f32] {
  &[
    // left column front
    0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    0.0, 1.0,
    1.0, 1.0,
    1.0, 0.0,

    // top rung front
    0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    0.0, 1.0,
    1.0, 1.0,
    1.0, 0.0,

    // middle rung front
    0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    0.0, 1.0,
    1.0, 1.0,
    1.0, 0.0,

    // left column back
    0.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0,

    // top rung back
    0.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0,

    // middle rung back
    0.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0,

    // top
    0.0, 0.0,
    1.0, 0.0,
    1.0, 1.0,
    0.0, 0.0,
    1.0, 1.0,
    0.0, 1.0,

    // top rung right
    0.0, 0.0,
    1.0, 0.0,
    1.0, 1.0,
    0.0, 0.0,
    1.0, 1.0,
    0.0, 1.0,

    // under top rung
    0.0, 0.0,
    0.0, 1.0,
    1.0, 1.0,
    0.0, 0.0,
    1.0, 1.0,
    1.0, 0.0,

    // between top rung and middle
    0.0, 0.0,
    1.0, 1.0,
    0.0, 1.0,
    0.0, 0.0,
    1.0, 0.0,
    1.0, 1.0,

    // top of middle rung
    0.0, 0.0,
    1.0, 1.0,
    0.0, 1.0,
    0.0, 0.0,
    1.0, 0.0,
    1.0, 1.0,

    // right of middle rung
    0.0, 0.0,
    1.0, 1.0,
    0.0, 1.0,
    0.0, 0.0,
    1.0, 0.0,
    1.0, 1.0,

    // bottom of middle rung.
    0.0, 0.0,
    0.0, 1.0,
    1.0, 1.0,
    0.0, 0.0,
    1.0, 1.0,
    1.0, 0.0,

    // right of bottom
    0.0, 0.0,
    1.0, 1.0,
    0.0, 1.0,
    0.0, 0.0,
    1.0, 0.0,
    1.0, 1.0,

    // bottom
    0.0, 0.0,
    0.0, 1.0,
    1.0, 1.0,
    0.0, 0.0,
    1.0, 1.0,
    1.0, 0.0,

    // left side
    0.0, 0.0,
    0.0, 1.0,
    1.0, 1.0,
    0.0, 0.0,
    1.0, 1.0,
    1.0, 0.0,
  ]
}

fn set_colors(gl: &WebGL2RenderingContext) {
  gl.buffer_data(BufferKind::Array, &[
    // left column front
    200,  70, 120,
    200,  70, 120,
    200,  70, 120,
    200,  70, 120,
    200,  70, 120,
    200,  70, 120,

    // top rung front
    200,  70, 120,
    200,  70, 120,
    200,  70, 120,
    200,  70, 120,
    200,  70, 120,
    200,  70, 120,

    // middle rung front
    200,  70, 120,
    200,  70, 120,
    200,  70, 120,
    200,  70, 120,
    200,  70, 120,
    200,  70, 120,

    // left column back
    80, 70, 200,
    80, 70, 200,
    80, 70, 200,
    80, 70, 200,
    80, 70, 200,
    80, 70, 200,

    // top rung back
    80, 70, 200,
    80, 70, 200,
    80, 70, 200,
    80, 70, 200,
    80, 70, 200,
    80, 70, 200,

    // middle rung back
    80, 70, 200,
    80, 70, 200,
    80, 70, 200,
    80, 70, 200,
    80, 70, 200,
    80, 70, 200,

    // top
    70, 200, 210,
    70, 200, 210,
    70, 200, 210,
    70, 200, 210,
    70, 200, 210,
    70, 200, 210,

    // top rung right
    200, 200, 70,
    200, 200, 70,
    200, 200, 70,
    200, 200, 70,
    200, 200, 70,
    200, 200, 70,

    // under top rung
    210, 100, 70,
    210, 100, 70,
    210, 100, 70,
    210, 100, 70,
    210, 100, 70,
    210, 100, 70,

    // between top rung and middle
    210, 160, 70,
    210, 160, 70,
    210, 160, 70,
    210, 160, 70,
    210, 160, 70,
    210, 160, 70,

    // top of middle rung
    70, 180, 210,
    70, 180, 210,
    70, 180, 210,
    70, 180, 210,
    70, 180, 210,
    70, 180, 210,

    // right of middle rung
    100, 70, 210,
    100, 70, 210,
    100, 70, 210,
    100, 70, 210,
    100, 70, 210,
    100, 70, 210,

    // bottom of middle rung.
    76, 210, 100,
    76, 210, 100,
    76, 210, 100,
    76, 210, 100,
    76, 210, 100,
    76, 210, 100,

    // right of bottom
    140, 210, 80,
    140, 210, 80,
    140, 210, 80,
    140, 210, 80,
    140, 210, 80,
    140, 210, 80,

    // bottom
    90, 130, 110,
    90, 130, 110,
    90, 130, 110,
    90, 130, 110,
    90, 130, 110,
    90, 130, 110,

    // left side
    160, 160, 220,
    160, 160, 220,
    160, 160, 220,
    160, 160, 220,
    160, 160, 220,
    160, 160, 220,
  ], DrawMode::Static)
}
