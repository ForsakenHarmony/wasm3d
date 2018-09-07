#![allow(unused)]

pub mod app;
pub mod webgl;

pub mod mesh;
pub mod renderer;

use cgmath::{
  perspective, Deg, EuclideanSpace, Matrix, Matrix4, Point3, Rad, SquareMatrix, Transform, Vector3,
};
use engine::app::App as WebApp;
use engine::app::*;
use engine::webgl::*;
use std::collections::HashMap;
use std::panic;
use std::rc::Rc;
use std::any::TypeId;

type Result<R> = ::std::result::Result<R, Box<::std::error::Error>>;

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

pub struct ShaderProgram {
  program: WebGLProgram,
  gl: Rc<WebGL2RenderingContext>,
  u_matrix: MatrixUniform,
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

pub struct Renderer {
  gl: Rc<WebGL2RenderingContext>,
  projection: Matrix4<f32>,
  size: (u32, u32),
  shaders: HashMap<TypeId, ShaderProgram>,
  active_shader: Option<TypeId>,
  shader_config: ShaderConfig,
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
      active_shader: None,
      shader_config,
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
      width,
      height,
      PixelFormat::Rgba,
      DataType::U8,
      pixels,
    );
    // gl.generate_mipmap(TextureKind::Texture2d);
    texture
  }

  pub fn create_mesh<V: VertexFormat>(&mut self, vertices: Vec<V>, indices: Option<Vec<u16>>) -> Mesh<V> where V: 'static {
    let program = self.shaders.entry(TypeId::of::<V>()).or_insert(ShaderProgram::new(Rc::clone(&self.gl), &self.shader_config));
    Mesh::new(&program, vertices, indices)
  }

  pub fn clear(&self, r: f32, g: f32, b: f32, a: f32) {
    self.gl.clear_color(r, g, b, a);
    self.gl.clear(BufferBit::Color);
    self.gl.clear(BufferBit::Depth);
  }

  fn start(&self) {
    self.gl.enable(Flag::CullFace);
    self.gl.enable(Flag::DepthTest);
  }

  pub fn render_mesh<V: VertexFormat>(&mut self, mesh: &Mesh<V>, transform: Matrix4<f32>) where V: 'static {
    let typeid = TypeId::of::<V>();
    let program = self.shaders.get_mut(&TypeId::of::<V>()).expect("There should be a program for a mesh that was previously created with it");
    if let Some(active) = self.active_shader {
      if active != typeid {
        program.use_program();
      }
    } else {
      program.use_program();
    }
    program.u_matrix.set(self.projection * transform);

    self.gl.bind_vertex_array(&mesh.vao);
    self.gl.draw_elements(Primitives::Triangles, mesh.vertices.len(), DataType::U16, 0);
  }
}

pub trait VertexFormat {
  type Buffers;

  fn create_buffers(program: &ShaderProgram, vertices: &Vec<Self>) -> Self::Buffers where Self: ::std::marker::Sized;
}

pub struct VertexPosTex {
  pub pos: [f32; 3],
  pub tex: [f32; 2],
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

pub struct Mesh<V: VertexFormat> {
  vertices: Vec<V>,
  indices: Vec<u16>,
  vao: WebGLVertexArrayObject,
  buffers: V::Buffers,
  index_buffer: WebGLBuffer,
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

pub trait State {
  fn new(renderer: &mut Renderer) -> Result<Self> where Self: ::std::marker::Sized;
  fn update(&mut self, delta: f32) -> Result<()>;
  fn render(&mut self, renderer: &mut Renderer) -> Result<()>;
  fn event(&mut self, event: AppEvent) -> Result<()> { Ok(()) }
}

pub fn run<T: State>(size: (u32, u32), title: &'static str) where T: 'static {
  let config = AppConfig::new(title, size);
  let web_app = WebApp::new(config);

  let gl = WebGL2RenderingContext::new(web_app.canvas());
  let gl = Rc::new(gl);

  let aspect = size.0 as f32 / size.1 as f32;

  let vert_code = include_str!("./shaders/vert.glsl");
  let frag_code = include_str!("./shaders/frag.glsl");

  let mut renderer = Renderer::new(Rc::clone(&gl), size, ShaderConfig::default());

  let mut state = T::new(&mut renderer).unwrap();

  let mut last = 0.0;
  web_app.run(move |app, t| {
    let t = t as f32;
    let delta = (t - last) / 1000.0;
    last = t;

    for event in app.events.borrow_mut().drain(..) {
      state.event(event);
    }

    state.update(delta);

    gl.viewport(0, 0, size.0, size.1);

    renderer.start();

    state.render(&mut renderer);
  });
}
