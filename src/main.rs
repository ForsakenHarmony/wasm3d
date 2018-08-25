#![feature(nll)]
#![recursion_limit = "512"]
// FIXME: remove this
#![allow(unused, unused_mut, dead_code)]

extern crate console_error_panic_hook;

#[macro_use]
extern crate stdweb;
#[macro_use]
extern crate stdweb_derive;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate cgmath;
extern crate image;
extern crate rand;

mod engine;
mod util;

use cgmath::{
  perspective, Deg, EuclideanSpace, Matrix, Matrix4, Point3, Rad, SquareMatrix, Transform, Vector3,
};
use engine::app::App as WebApp;
use engine::app::*;
use engine::webgl::*;
use std::collections::HashMap;
use std::panic;
use std::rc::Rc;

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

struct VBO {
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
      .buffer_data(BufferKind::Array, data, DrawMode::Static);
  }

  pub fn set_data_bytes(&self, data: &[u8]) {
    self.gl.bind_buffer(BufferKind::Array, &self.buffer);
    self
      .gl
      .buffer_data_bytes(BufferKind::Array, data, DrawMode::Static);
  }
}

struct ShaderProgram {
  program: WebGLProgram,
  gl: Rc<WebGL2RenderingContext>,
}

impl ShaderProgram {
  pub fn new(
    gl: Rc<WebGL2RenderingContext>,
    vert_code: &'static str,
    frag_code: &'static str,
  ) -> Self {
    let vert_shader = create_shader(&gl, ShaderKind::Vertex, vert_code);
    let frag_shader = create_shader(&gl, ShaderKind::Fragment, frag_code);
    let program = create_program(&gl, &vert_shader, &frag_shader);

    ShaderProgram { program, gl }
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

fn create_texture(
  gl: &WebGL2RenderingContext,
  width: u16,
  height: u16,
  pixels: &[u8],
) -> WebGLTexture {
  let texture = gl.create_texture();
  gl.bind_texture(&texture);
  gl.tex_parameteri(
    TextureParameter::TextureMinFilter,
    TextureMinFilter::Nearest as i32,
  );
  gl.tex_parameteri(
    TextureParameter::TextureMagFilter,
    TextureMagFilter::Nearest as i32,
  );
  gl.tex_image2d(
    TextureBindPoint::Texture2d,
    0,
    width,
    height,
    PixelFormat::Rgba,
    DataType::U8,
    pixels,
  );
  // gl.generate_mipmap(TextureKind::Texture2d);
  texture
}

fn load_image(buffer: &[u8]) -> Result<(u16, u16, Vec<u8>), Box<std::error::Error>> {
  let img = image::load_from_memory(buffer)?.to_rgba();
  Ok((img.width() as u16, img.height() as u16, img.into_raw()))
}

struct MatrixUniform {
  uniform: WebGLUniformLocation,
  gl: Rc<WebGL2RenderingContext>,
}

trait Uniform {
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

fn create_and_bind_buffer(
  gl: &WebGL2RenderingContext,
  size: AttributeSize,
  data: &[f32],
  loc: u32,
) -> WebGLBuffer {
  let buffer = gl.create_buffer();
  gl.bind_buffer(BufferKind::Array, &buffer);
  gl.buffer_data(BufferKind::Array, data, DrawMode::Static);
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
    let projection_matrix = perspective(Deg(60.0), aspect, 1.0, 2000.0);

    Camera {
      proj: perspective(fov, aspect, near, far),
      view,
    }
  }
}

struct Renderer {
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  camera: Camera,
  size: (u32, u32),
}

impl Renderer {
  pub fn new(
    gl: WebGL2RenderingContext,
    vert_code: &'static str,
    frag_code: &'static str,
    camera: Camera,
  ) -> Self {
    let vert_shader = create_shader(&gl, ShaderKind::Vertex, vert_code);
    let frag_shader = create_shader(&gl, ShaderKind::Fragment, frag_code);
    let program = create_program(&gl, &vert_shader, &frag_shader);

    Renderer {
      gl,
      program,
      camera,
      size: (1280, 720),
    }
  }

  fn start(&self) {
    self.gl.use_program(&self.program);
  }
}

trait VertexFormat {}

struct VertexPosTex {
  pos: [f32; 3],
  tex: [f32; 2],
}
impl VertexFormat for VertexPosTex {}

struct Mesh<V: VertexFormat> {
  vertices: Vec<V>,
  indices: Vec<u16>,
}

impl<V: VertexFormat> Mesh<V> {
  pub fn new(vertices: Vec<V>, indices: Vec<u16>) -> Self {
    Mesh { vertices, indices }
  }

  pub fn new_no_indices(vertices: Vec<V>) -> Self {
    let indices = (0u16..vertices.len() as u16 - 1u16).collect();

    Mesh { vertices, indices }
  }
}

struct GpuMesh {}

struct App {
  renderer: Renderer,
  gl: Rc<WebGL2RenderingContext>,
}

// impl App {
//   pub fn new(gl: WebGL2RenderingContext) -> Self {
//     App {
//       gl: Rc::new(gl),
//     }
//   }
// }

// trait State {
//   fn update(&mut self, events: Vec<AppEvent>) -> StateChange;
//   fn render(&mut self, renderer: &Renderer);
// }

// enum StateChange {
//   Pop,
//   Push(Box<State>),
//   Replace(Box<State>),
//   None
// }

// struct AssetLoader {
//   assets: Vec<(&'static str, Vec<u8>)>,
// }

// impl AssetLoader {
//   pub fn new(assets: &[&str]) -> Self {
//     AssetLoader {
//       assets: assets.iter().map(|path| FileSystem::open(path)).collect()
//     }
//   }
// }

// impl State for AssetLoader {
//   fn update(&mut self, _: Vec<AppEvent>) -> StateChange {

//   }

//   fn render(&mut self, _: &Renderer) {}
// }

// struct Context<'a> {
//   renderer: &'a Renderer,
//   app: &'a App,
//   delta: f32,
// }

// struct App {
//   stack: Vec<Box<State>>,
//   assets: HashMap<&'static str, Vec<u8>>,
//   renderer: Renderer,
// }

// impl App {
//   pub fn new(renderer: Renderer, initial_state: State, assets: &[&'static str]) -> Self {
//     let stack = Vec::new();
//     stack.push(initial_state);

//     if assets.len() > 0 {
//       stack.push(AssetLoader::new(assets));
//     }

//     App {
//       stack,
//       assets: HashMap::new(),
//       renderer,
//     }
//   }

//   pub fn run(&mut self) {

//   }

//   // return is "should continue"
//   fn update(&mut self) -> bool {
//     let state = self.stack.last().expect("at this point we should always at least have one state");
//     match state.update(&self.renderer) {
//       Pop => {
//         self.stack.pop();
//         if self.stack.len() == 0 {
//           return false,
//         }
//       },
//       Push(state) => {
//         self.stack.push(state);
//       },
//       Replace(state) => {
//         self.stack.pop();
//         self.stack.push(state);
//       }
//       None => {},
//     }
//     true
//   }
// }

#[cfg(target_arch = "wasm32")]
pub fn main() -> Result<(), Box<std::error::Error>> {
  panic::set_hook(Box::new(console_error_panic_hook::hook));

  let thing = 3;

  let size = (1280, 720);
  let config = AppConfig::new("Test", size);
  let app = WebApp::new(config);

  let img = load_image(include_bytes!("../static/f-texture.png"))?;

  let gl = WebGL2RenderingContext::new(app.canvas());
  let gl = Rc::new(gl);

  let vert_code = include_str!("../shaders/vert.glsl");
  let frag_code = include_str!("../shaders/frag.glsl");

  let shader_program = ShaderProgram::new(Rc::clone(&gl), vert_code, frag_code);
  let vao = shader_program.create_vertex_array();
  let pos_buffer =
    shader_program.create_buffer(AttributeSize::Three, "a_position", DataType::Float);
  let tex_buffer = shader_program.create_buffer(AttributeSize::Two, "a_texcoord", DataType::Float);
  let u_matrix = shader_program.create_uniform::<MatrixUniform>("u_matrix");

  pos_buffer.set_data(get_geometry().as_slice());
  tex_buffer.set_data(get_texcoords());

  // look up data locations
  // let pos_attr_loc = gl.get_attrib_location(&program, "a_position").unwrap();
  // let tex_coord_loc = gl.get_attrib_location(&program, "a_texcoord").unwrap();

  // let u_matrix = create_uniform::<MatrixUniform>(&gl, &program, "u_matrix");

  // look up uniforms
  //  let matrix_loc = gl.get_uniform_location(&program, "u_matrix").unwrap();

  // let vao = gl.create_vertex_array();
  // gl.bind_vertex_array(&vao);

  // create_and_bind_buffer(&gl, AttributeSize::Three, get_geometry().as_slice(), pos_attr_loc);
  // create_and_bind_buffer(&gl, AttributeSize::Two, get_texcoords(), tex_coord_loc);

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

  let texture = create_texture(&gl, img.0, img.1, img.2.as_slice());

  //  let texture = gl.create_texture();
  //  gl.active_texture(TextureIndex::Texture0);
  //  gl.bind_texture(&texture);
  //
  //  gl.tex_parameteri(TextureParameter::TextureWrapS, TextureWrap::ClampToEdge as i32);
  //  gl.tex_parameteri(TextureParameter::TextureWrapT, TextureWrap::ClampToEdge as i32);
  //
  //  gl.tex_image2d(TextureBindPoint::Texture2d, 0, 240, 180, PixelFormat::Rgba, DataType::U8, img.raw_pixels().as_slice());

  let mut translation = Vector3::from((-150.0, 0.0, -660.0));
  let mut scale = Vector3::from((1.0, 1.0, 1.0));
  let mut angle = 0.0;

  fn to_rad(angle: f32) -> f32 {
    (360.0 - angle) * std::f32::consts::PI / 180.0
  }

  let mut last = 0.0;
  app.run(move |app: &mut WebApp, t: f64| {
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

    shader_program.use_program();

    // gl.use_program(&program);

    gl.bind_vertex_array(&vao);

    let num_fs = 5;
    let radius = 200.0;

    let aspect = size.0 as f32 / size.1 as f32;

    let f_pos = Point3::new(radius, 0.0, 0.0);

    let projection_matrix = perspective(Deg(60.0), aspect, 1.0, 2000.0);

    let camera_matrix = Matrix4::from_angle_y(radian_angle)
      * Matrix4::from_translation(Vector3::new(0.0, 50.0, radius * 1.5));

    let cam_pos = camera_matrix.transform_point(Point3::origin());

    let view_matrix = Matrix4::look_at(cam_pos, f_pos, Vector3::unit_y());

    let view_projection_matrix = projection_matrix * view_matrix;

    for i in 0..num_fs {
      let angle = i as f32 * ::std::f32::consts::PI * 2.0 / num_fs as f32;

      let x = angle.cos() * radius;
      let z = angle.sin() * radius;

      let matrix = view_projection_matrix * Matrix4::from_translation(Vector3::new(x, 0.0, z));

      u_matrix.set(matrix);

      gl.draw_arrays(Primitives::Triangles, 16 * 6);
    }
  });

  Ok(())
}

fn get_geometry() -> Vec<f32> {

  let arr = [
    // left column front
    0.0,  0.0,   0.0,
    0.0,  150.0, 0.0,
    30.0, 0.0,   0.0,
    0.0,  150.0, 0.0,
    30.0, 150.0, 0.0,
    30.0, 0.0,   0.0,
    // top rung front
    30.0, 0.0, 0.0, 30.0, 30.0, 0.0, 100.0, 0.0, 0.0, 30.0, 30.0, 0.0, 100.0, 30.0, 0.0, 100.0,
    0.0, 0.0, // middle rung front
    30.0, 60.0, 0.0, 30.0, 90.0, 0.0, 67.0, 60.0, 0.0, 30.0, 90.0, 0.0, 67.0, 90.0, 0.0, 67.0,
    60.0, 0.0, // left column back
    0.0, 0.0, 30.0, 30.0, 0.0, 30.0, 0.0, 150.0, 30.0, 0.0, 150.0, 30.0, 30.0, 0.0, 30.0, 30.0,
    150.0, 30.0, // top rung back
    30.0, 0.0, 30.0, 100.0, 0.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 100.0, 0.0, 30.0, 100.0,
    30.0, 30.0, // middle rung back
    30.0, 60.0, 30.0, 67.0, 60.0, 30.0, 30.0, 90.0, 30.0, 30.0, 90.0, 30.0, 67.0, 60.0, 30.0, 67.0,
    90.0, 30.0, // top
    0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 100.0, 0.0, 30.0, 0.0, 0.0, 0.0, 100.0, 0.0, 30.0, 0.0, 0.0,
    30.0, // top rung right
    100.0, 0.0, 0.0, 100.0, 30.0, 0.0, 100.0, 30.0, 30.0, 100.0, 0.0, 0.0, 100.0, 30.0, 30.0,
    100.0, 0.0, 30.0, // under top rung
    30.0, 30.0, 0.0, 30.0, 30.0, 30.0, 100.0, 30.0, 30.0, 30.0, 30.0, 0.0, 100.0, 30.0, 30.0,
    100.0, 30.0, 0.0, // between top rung and middle
    30.0, 30.0, 0.0, 30.0, 60.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 0.0, 30.0, 60.0, 0.0, 30.0,
    60.0, 30.0, // top of middle rung
    30.0, 60.0, 0.0, 67.0, 60.0, 30.0, 30.0, 60.0, 30.0, 30.0, 60.0, 0.0, 67.0, 60.0, 0.0, 67.0,
    60.0, 30.0, // right of middle rung
    67.0, 60.0, 0.0, 67.0, 90.0, 30.0, 67.0, 60.0, 30.0, 67.0, 60.0, 0.0, 67.0, 90.0, 0.0, 67.0,
    90.0, 30.0, // bottom of middle rung.
    30.0, 90.0, 0.0, 30.0, 90.0, 30.0, 67.0, 90.0, 30.0, 30.0, 90.0, 0.0, 67.0, 90.0, 30.0, 67.0,
    90.0, 0.0, // right of bottom
    30.0, 90.0, 0.0, 30.0, 150.0, 30.0, 30.0, 90.0, 30.0, 30.0, 90.0, 0.0, 30.0, 150.0, 0.0, 30.0,
    150.0, 30.0, // bottom
    0.0, 150.0, 0.0, 0.0, 150.0, 30.0, 30.0, 150.0, 30.0, 0.0, 150.0, 0.0, 30.0, 150.0, 30.0, 30.0,
    150.0, 0.0, // left side
    0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 0.0, 150.0, 30.0, 0.0, 0.0, 0.0, 0.0, 150.0, 30.0, 0.0, 150.0,
    0.0,
  ];

  let matrix = Matrix4::from_angle_x(Deg(180.0))
    * Matrix4::from_translation(Vector3::new(-50.0, -75.0, -15.0));

  let mut vec = Vec::<f32>::new();

  for coord in arr.chunks(3) {
    let out: [f32; 3] = matrix
      .transform_point([coord[0], coord[1], coord[2]].into())
      .into();
    vec.extend_from_slice(&out);
  }

  vec
}

fn get_texcoords() -> &'static [f32] {
  &[
    // left column front
    0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, // top rung front
    0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, // middle rung front
    0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, // left column back
    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, // top rung back
    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, // middle rung back
    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, // top
    0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, // top rung right
    0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, // under top rung
    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
    // between top rung and middle
    0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, // top of middle rung
    0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, // right of middle rung
    0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, // bottom of middle rung.
    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, // right of bottom
    0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, // bottom
    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, // left side
    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
  ]
}

fn set_colors(gl: &WebGL2RenderingContext) {
  gl.buffer_data_bytes(
    BufferKind::Array,
    &[
      // left column front
      200, 70, 120, 200, 70, 120, 200, 70, 120, 200, 70, 120, 200, 70, 120, 200, 70, 120,
      // top rung front
      200, 70, 120, 200, 70, 120, 200, 70, 120, 200, 70, 120, 200, 70, 120, 200, 70, 120,
      // middle rung front
      200, 70, 120, 200, 70, 120, 200, 70, 120, 200, 70, 120, 200, 70, 120, 200, 70, 120,
      // left column back
      80, 70, 200, 80, 70, 200, 80, 70, 200, 80, 70, 200, 80, 70, 200, 80, 70, 200,
      // top rung back
      80, 70, 200, 80, 70, 200, 80, 70, 200, 80, 70, 200, 80, 70, 200, 80, 70, 200,
      // middle rung back
      80, 70, 200, 80, 70, 200, 80, 70, 200, 80, 70, 200, 80, 70, 200, 80, 70, 200, // top
      70, 200, 210, 70, 200, 210, 70, 200, 210, 70, 200, 210, 70, 200, 210, 70, 200, 210,
      // top rung right
      200, 200, 70, 200, 200, 70, 200, 200, 70, 200, 200, 70, 200, 200, 70, 200, 200, 70,
      // under top rung
      210, 100, 70, 210, 100, 70, 210, 100, 70, 210, 100, 70, 210, 100, 70, 210, 100, 70,
      // between top rung and middle
      210, 160, 70, 210, 160, 70, 210, 160, 70, 210, 160, 70, 210, 160, 70, 210, 160, 70,
      // top of middle rung
      70, 180, 210, 70, 180, 210, 70, 180, 210, 70, 180, 210, 70, 180, 210, 70, 180, 210,
      // right of middle rung
      100, 70, 210, 100, 70, 210, 100, 70, 210, 100, 70, 210, 100, 70, 210, 100, 70, 210,
      // bottom of middle rung.
      76, 210, 100, 76, 210, 100, 76, 210, 100, 76, 210, 100, 76, 210, 100, 76, 210, 100,
      // right of bottom
      140, 210, 80, 140, 210, 80, 140, 210, 80, 140, 210, 80, 140, 210, 80, 140, 210, 80,
      // bottom
      90, 130, 110, 90, 130, 110, 90, 130, 110, 90, 130, 110, 90, 130, 110, 90, 130, 110,
      // left side
      160, 160, 220, 160, 160, 220, 160, 160, 220, 160, 160, 220, 160, 160, 220, 160, 160, 220,
    ],
    DrawMode::Static,
  )
}
