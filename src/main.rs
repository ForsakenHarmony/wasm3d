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
extern crate gltf;
#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate engine_codegen;

mod engine;
mod util;

use cgmath::{
  perspective, Deg, EuclideanSpace, Matrix, Matrix4, Point3, Rad, SquareMatrix, Transform, Vector3,
};
use gltf::Gltf;

use engine::{
  run,
  State,
  renderer::{Renderer, Camera},
  mesh::{Mesh, VertexPosTex},
  webgl::{WebGLTexture},
};
use std::collections::HashMap;

type Result<R> = std::result::Result<R, Box<std::error::Error>>;

fn load_image(buffer: &[u8]) -> Result<(u16, u16, Vec<u8>)> {
  let img = image::load_from_memory(buffer)?.to_rgba();
  Ok((img.width() as u16, img.height() as u16, img.into_raw()))
}

//fn load_gltf(buffer: &[u8]) -> Result<(gltf::Document, Vec<gltf::buffer::Data>, Vec<gltf::image::Data>)> {
//  let Gltf { document, blob } = Gltf::from_slice(buffer)?;
//  let buffer_data = gltf::import::import_buffer_data(&document, base, blob)?;
//  let image_data = gltf::import::import_image_data(&document, base, &buffer_data)?;
//  Ok((document, buffer_data, image_data))
//}

struct GameState {
  f: Mesh<VertexPosTex>,
  texture: WebGLTexture,
  angle: f32,
  camera: Camera,
}

impl State for GameState {
  fn new(renderer: &mut Renderer) -> Result<Self> {
    let vertices = VertexPosTex {
      pos: get_geometry(),
      tex: get_texcoords(),
    };

    let img = load_image(include_bytes!("../static/f-texture.png"))?;

    Ok(GameState {
      f: renderer.create_mesh(vertices, None),
      texture: renderer.create_texture(img.2.as_slice(), img.0, img.1),
      angle: 0.0,
      camera: Camera::perspective(Deg(60.0), renderer.aspect(), 1.0, 2000.0, Point3::origin()),
    })
  }

  fn update(&mut self, delta: f64) -> Result<()> {
    self.angle = (self.angle + 10.0 * delta as f32) % 360.0;
    Ok(())
  }

  fn render(&mut self, renderer: &mut Renderer) -> Result<()> {
    renderer.clear(0.0, 0.0, 0.0, 0.0);

    let num_fs = 5;
    let radius = 200.0;

    let f_pos = Point3::new(radius, 0.0, 0.0);

    let camera_matrix = Matrix4::from_angle_y(Deg(self.angle))
        * Matrix4::from_translation(Vector3::new(0.0, 50.0, radius * 1.5));
    let cam_pos = camera_matrix.transform_point(Point3::origin());

    self.camera.set_pos(cam_pos);
    self.camera.look_at(f_pos);
    self.camera.update();

    renderer.set_projection(self.camera.combined());

    for i in 0..num_fs {
      let angle = i as f32 * ::std::f32::consts::PI * 2.0 / num_fs as f32;

      let x = angle.cos() * radius;
      let z = angle.sin() * radius;

      let transform = Matrix4::from_translation(Vector3::new(x, 0.0, z));

      renderer.render_mesh(&self.f, transform);
    }

    Ok(())
  }
}

fn log<T: Into<String>>(msg: T) {
  js! {@(no_return)
    console.log(@{msg.into()});
  }
}

#[cfg(target_arch = "wasm32")]
pub fn main() {
  std::panic::set_hook(Box::new(|info: &std::panic::PanicInfo| {
    js! {@(no_return)
      console.error(@{format!("{:#?}", info)});
    }
  }));

//  let (document, buffers, images) = load_gltf(include_bytes!("../static/fox.glb")).unwrap();
  let (document, buffers, images) = ::gltf::import_slice(include_bytes!("../static/fox.glb")).unwrap();

  log(format!("{:#?}", buffers.len()));
  log(format!("{:#?}", images.len()));

  let thing: ::gltf::Semantic;
  let thing: ::gltf::Accessor;
  let thing: ::gltf::Buffer;
  let thing: ::gltf::buffer::View;

  for buffer in document.buffers() {
    log(format!("Buffer: {:#?} {:#?} {:#?} {:#?}", buffer.name(), buffer.length(), buffer.index(), buffer.extras()));
  }

  for view in document.views() {
    log(format!("View: {:#?} {:#?} {:#?} {:#?} {:#?}", view.name(), view.length(), view.index(), view.offset(), view.extras()));
  }

  let accessors = document.accessors().map(|acc| {
    let view = acc.view();
    let buffer = view.buffer();

    let offset = view.offset();
    let length = view.length();

    let buffer_slice = &buffers[buffer.index()].0[offset..offset + length];

    use ::gltf::json::accessor::ComponentType::*;
    use ::gltf::json::accessor::Type::*;

    let data_type = acc.data_type();

    let type_size = match data_type {
      I8 => 1,
      U8 => 1,
      I16 => 2,
      U16 => 2,
      U32 => 4,
      F32 => 4,
    };

    #[derive(Copy, Clone, Debug)]
    enum Types {
      I8(i8),
      U8(u8),
      I16(i16),
      U16(u16),
      F32(f32),
      U32(u32),
    }

    fn cast<T: Copy>(chunk: &[u8]) -> T {
      unsafe { *(chunk.as_ptr() as *const T) }
    }

    let typed_buffer = buffer_slice.chunks(type_size).map(|chunk| match data_type {
      I8 => Types::I8(cast(chunk)),
      U8 => Types::U8(cast(chunk)),
      I16 => Types::I16(cast(chunk)),
      U16 => Types::U16(cast(chunk)),
      U32 => Types::U32(cast(chunk)),
      F32 => Types::F32(cast(chunk)),
    }).collect::<Vec<_>>();

    #[derive(Debug)]
    enum Container<T: Copy> {
      Scalar(T),
      Vec2([T; 2]),
      Vec3([T; 3]),
      Vec4([T; 4]),
      Mat2([[T; 2]; 2]),
      Mat3([[T; 3]; 3]),
      Mat4([[T; 4]; 4]),
    }

    let dimensions = acc.dimensions();
    let container_size = match dimensions {
      Scalar => 1,
      Vec2 => 2,
      Vec3 => 3,
      Vec4 => 4,
      Mat2 => 4,
      Mat3 => 9,
      Mat4 => 16,
    };

    let containers = typed_buffer.chunks(container_size).map(|chunk| match dimensions {
      Scalar => Container::Scalar(chunk[0]),
      Vec2 => Container::Vec2([chunk[0], chunk[1]]),
      Vec3 => Container::Vec3([chunk[0], chunk[1], chunk[2]]),
      Vec4 => Container::Vec4([chunk[0], chunk[1], chunk[2], chunk[3]]),
      Mat2 => Container::Mat2([[chunk[0], chunk[1]], [chunk[2], chunk[3]]]),
      Mat3 => Container::Mat3([[chunk[0], chunk[1], chunk[2]], [chunk[3], chunk[4], chunk[5]], [chunk[6], chunk[7], chunk[8]]]),
      Mat4 => Container::Mat4([[chunk[0], chunk[1], chunk[2], chunk[3]], [chunk[4], chunk[5], chunk[6], chunk[7]], [chunk[8], chunk[9], chunk[10], chunk[11]], [chunk[12], chunk[13], chunk[14], chunk[15]]]),
    }).collect::<Vec<_>>();

    containers
  }).collect::<Vec<_>>();

  log(format!("Accessors: {:#?}", accessors));

  for acc in document.accessors() {
    log(format!("Accessor: {:#?} {:#?} {:#?} {:#?}", acc.name(), acc.index(), acc.offset(), acc.data_type()));
  }

  for mesh in document.meshes() {
    for prim in mesh.primitives() {
      let position_accessor = prim.get(&::gltf::Semantic::Positions).unwrap();

      //      log(format!("{:#?}", prim));
      use std::iter::FromIterator;
      let map: HashMap<::gltf::Semantic, ::gltf::Accessor> = HashMap::from_iter(prim.attributes());

      for (sem, acc) in prim.attributes() {
        let view = acc.view();
        let buffer = view.buffer();
        log(format!("Semantic: {:#?} \nAccessor: {:#?} {:#?} {:#?} {:#?} {:#?}\nView: {:#?} {:#?} {:#?} {:#?} \nBuffer: {:#?}", sem, acc.index(), acc.offset(), acc.data_type(), acc.dimensions(), acc.count(), view.index(), view.offset(), view.length(), view.target(), buffer.index()));
      }
    }
  }

//  log(format!("{:?}", gltf));

  run::<GameState>((1280, 720), "Test");
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

fn get_texcoords() -> Vec<f32> {
  vec![
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

//fn set_colors(gl: &WebGL2RenderingContext) {
//  gl.buffer_data_u8(
//    BufferKind::Array,
//    &[
//      // left column front
//      200, 70, 120, 200, 70, 120, 200, 70, 120, 200, 70, 120, 200, 70, 120, 200, 70, 120,
//      // top rung front
//      200, 70, 120, 200, 70, 120, 200, 70, 120, 200, 70, 120, 200, 70, 120, 200, 70, 120,
//      // middle rung front
//      200, 70, 120, 200, 70, 120, 200, 70, 120, 200, 70, 120, 200, 70, 120, 200, 70, 120,
//      // left column back
//      80, 70, 200, 80, 70, 200, 80, 70, 200, 80, 70, 200, 80, 70, 200, 80, 70, 200,
//      // top rung back
//      80, 70, 200, 80, 70, 200, 80, 70, 200, 80, 70, 200, 80, 70, 200, 80, 70, 200,
//      // middle rung back
//      80, 70, 200, 80, 70, 200, 80, 70, 200, 80, 70, 200, 80, 70, 200, 80, 70, 200, // top
//      70, 200, 210, 70, 200, 210, 70, 200, 210, 70, 200, 210, 70, 200, 210, 70, 200, 210,
//      // top rung right
//      200, 200, 70, 200, 200, 70, 200, 200, 70, 200, 200, 70, 200, 200, 70, 200, 200, 70,
//      // under top rung
//      210, 100, 70, 210, 100, 70, 210, 100, 70, 210, 100, 70, 210, 100, 70, 210, 100, 70,
//      // between top rung and middle
//      210, 160, 70, 210, 160, 70, 210, 160, 70, 210, 160, 70, 210, 160, 70, 210, 160, 70,
//      // top of middle rung
//      70, 180, 210, 70, 180, 210, 70, 180, 210, 70, 180, 210, 70, 180, 210, 70, 180, 210,
//      // right of middle rung
//      100, 70, 210, 100, 70, 210, 100, 70, 210, 100, 70, 210, 100, 70, 210, 100, 70, 210,
//      // bottom of middle rung.
//      76, 210, 100, 76, 210, 100, 76, 210, 100, 76, 210, 100, 76, 210, 100, 76, 210, 100,
//      // right of bottom
//      140, 210, 80, 140, 210, 80, 140, 210, 80, 140, 210, 80, 140, 210, 80, 140, 210, 80,
//      // bottom
//      90, 130, 110, 90, 130, 110, 90, 130, 110, 90, 130, 110, 90, 130, 110, 90, 130, 110,
//      // left side
//      160, 160, 220, 160, 160, 220, 160, 160, 220, 160, 160, 220, 160, 160, 220, 160, 160, 220,
//    ],
//    DrawMode::Static,
//  )
//}
