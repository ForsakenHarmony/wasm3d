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

pub struct MeshRef(usize);

pub struct Renderer {
  pub(crate) gl: Rc<WebGL2RenderingContext>,
  projection: Matrix4<f32>,
  size: (u32, u32),
  shaders: HashMap<TypeId, ShaderProgram>,
  active_shader: Option<TypeId>,
  shader_config: ShaderConfig,
  meshes: Vec<Mesh<VertexFormat>>,
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
      active_shader: None,
      shader_config,
      meshes: Vec::new(),
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

  pub fn create_mesh<V: VertexFormat>(&mut self, vertices: V, indices: Option<Vec<u16>>) -> Mesh<V> where V: Sized + 'static {
    // get the shader program for the vertex type, or create one
    let program = self.shaders.entry(TypeId::of::<V>()).or_insert(ShaderProgram::new::<V>(Rc::clone(&self.gl), &self.shader_config));
    let mesh = Mesh::new(&program, vertices, indices);
//    self.meshes.push(mesh);
//    MeshRef(self.meshes.len() - 1)
    mesh
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

  pub(crate) fn exec(&self) {

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

    mesh.vao.bind(|| {
      self.gl.draw_elements(Primitives::Triangles, mesh.indices.len(), DataType::U16, 0);
    });
  }
}
