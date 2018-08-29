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
  u_matrix: MatrixUniform,
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

struct Camera {
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

struct Renderer {
  gl: Rc<WebGL2RenderingContext>,
  program: ShaderProgram,
  camera: Camera,
  size: (u32, u32),
}

impl Renderer {
  pub fn new(
    gl: Rc<WebGL2RenderingContext>,
    vert_code: &'static str,
    frag_code: &'static str,
    camera: Camera,
  ) -> Self {
    let program = ShaderProgram::new(Rc::clone(&gl), vert_code, frag_code);

    Renderer {
      gl,
      program,
      camera,
      size: (1280, 720),
    }
  }

  pub fn create_mesh<V: VertexFormat>(&self, vertices: Vec<V>, indices: Option<Vec<u16>>) -> Mesh<V> {
    Mesh::new(&self.program, vertices, indices)
  }

  fn start(&self) {
    self.program.use_program();
  }

  pub fn render_mesh<V: VertexFormat>(&mut self, mesh: &Mesh<V>, translation: Vector3<f32>) {
    self.gl.bind_vertex_array(&mesh.vao);

    self.program.u_matrix.set(self.camera.view_proj * Matrix4::from_translation(translation));

    self.gl.draw_arrays(Primitives::Triangles,  mesh.vertices.len());
  }
}

trait VertexFormat {
  type Buffers;

  fn create_buffers(program: &ShaderProgram, vertices: &Vec<Self>) -> Self::Buffers where Self: std::marker::Sized;
}

struct VertexPosTex {
  pos: [f32; 3],
  tex: [f32; 2],
}

impl VertexFormat for VertexPosTex {
  type Buffers = (VBO, VBO);

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

struct Mesh<V: VertexFormat> {
  vertices: Vec<V>,
  indices: Vec<u16>,
  vao: WebGLVertexArrayObject,
  buffers: V::Buffers,
}

impl<V: VertexFormat> Mesh<V> {
  pub fn new(program: &ShaderProgram, vertices: Vec<V>, indices: Option<Vec<u16>>) -> Self {
    let indices = indices.unwrap_or((0u16..vertices.len() as u16 - 1u16).collect());

    let vao = program.create_vertex_array();
    let buffers = V::create_buffers(program, &vertices);

    Mesh {
      vertices,
      indices,
      vao,
      buffers,
    }
  }
}

struct App {
  renderer: Renderer,
  gl: Rc<WebGL2RenderingContext>,
}

fn panic_hook(info: &std::panic::PanicInfo) {
  js! {@(no_return)
    console.error(@{info.to_string()});
  }
}

#[cfg(target_arch = "wasm32")]
pub fn main() -> Result<(), Box<std::error::Error>> {
  std::panic::set_hook(Box::new(panic_hook));

  let thing = 3;

  let size = (1280, 720);
  let config = AppConfig::new("Test", size);
  let app = WebApp::new(config);

  let img = load_image(include_bytes!("../static/f-texture.png"))?;

  let gl = WebGL2RenderingContext::new(app.canvas());
  let gl = Rc::new(gl);

  let aspect = size.0 as f32 / size.1 as f32;

  let vert_code = include_str!("../shaders/vert.glsl");
  let frag_code = include_str!("../shaders/frag.glsl");

  let camera = Camera::perspective(Deg(60.0), aspect, 1.0, 2000.0, Point3::origin());
  let mut renderer = Renderer::new(Rc::clone(&gl), vert_code, frag_code, camera);

  let vertices = get_geometry().chunks(3).zip(get_texcoords().chunks(2)).map(|(geom, tex)| {
    VertexPosTex {
      pos: [geom[0], geom[1], geom[2]],
      tex: [tex[0], tex[1]],
    }
  }).collect();

  let f = renderer.create_mesh(vertices, None);

//  let shader_program = ShaderProgram::new(Rc::clone(&gl), vert_code, frag_code);
//  let vao = shader_program.create_vertex_array();
//  let pos_buffer =
//    shader_program.create_buffer(AttributeSize::Three, "a_position", DataType::Float);
//  let tex_buffer = shader_program.create_buffer(AttributeSize::Two, "a_texcoord", DataType::Float);
//  let u_matrix = shader_program.create_uniform::<MatrixUniform>("u_matrix");
//
//  pos_buffer.set_data(get_geometry().as_slice());
//  tex_buffer.set_data(get_texcoords());

  // look up data locations
  // let pos_attr_loc = gl.get_attrib_location(&program, "a_position").unwrap();
  // let tex_coord_loc = gl.get_attrib_location(&program, "a_texcoord").unwrap();

  // let u_matrix = create_uniform::<MatrixUniform>(&gl, &program, "u_matrix");

  // look up uniforms
  //  let matrix_loc = gl.get_uniform_location(&program, "u_matrix").unwrap();

  let texture = create_texture(&gl, img.0, img.1, img.2.as_slice());

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

    renderer.start();

    let num_fs = 5;
    let radius = 200.0;

    let aspect = size.0 as f32 / size.1 as f32;

    let f_pos = Point3::new(radius, 0.0, 0.0);

    let projection_matrix = perspective(Deg(60.0), aspect, 1.0, 2000.0);

    let camera_matrix = Matrix4::from_angle_y(radian_angle)
        * Matrix4::from_translation(Vector3::new(0.0, 50.0, radius * 1.5));
    let cam_pos = camera_matrix.transform_point(Point3::origin());

    renderer.camera.set_pos(cam_pos);
    renderer.camera.look_at(f_pos);
    renderer.camera.update();

    for i in 0..num_fs {
      let angle = i as f32 * ::std::f32::consts::PI * 2.0 / num_fs as f32;

      let x = angle.cos() * radius;
      let z = angle.sin() * radius;

//      let matrix = view_projection_matrix * Matrix4::from_translation(Vector3::new(x, 0.0, z));
//
//      u_matrix.set(matrix);

      renderer.render_mesh(&f, Vector3::new(x, 0.0, z));

//      gl.draw_arrays(Primitives::Triangles, 16 * 6);
    }
  });

  Ok(())
}

fn get_geometry() -> Vec<f32> {
  let arr = [
    // left column front
    0.0, 0.0, 0.0,
    0.0, 150.0, 0.0,
    30.0, 0.0, 0.0,
    0.0, 150.0, 0.0,
    30.0, 150.0, 0.0,
    30.0, 0.0, 0.0,
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
