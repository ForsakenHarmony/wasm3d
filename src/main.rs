#![feature(nll)]
#![recursion_limit = "512"]
// FIXME: remove this
#![allow(unused, unused_mut, dead_code)]

#[macro_use]
extern crate stdweb;
#[macro_use]
extern crate engine_codegen;
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

mod engine;
mod util;

use std::collections::HashMap;

use cgmath::{perspective, Deg, EuclideanSpace, Matrix, Matrix4, Point3, Rad, SquareMatrix, Transform, Vector3, Zero, Quaternion, Euler, Rotation};
use gltf::Gltf;
use rand::RngCore;
use stdweb::js;

use engine::{
  run,
  State,
  renderer::{Renderer, Camera, MeshRef},
  mesh::{Mesh, VertexPosTex, VertexPosCol, VertexFormat},
  webgl::WebGLTexture,
};
use crate::engine::app::AppEvent;
use stdweb::web::event::MouseButton;
use crate::engine::Ctx;
use crate::engine::input::Button;
use gltf::iter::Buffers;

type Result<R> = std::result::Result<R, Box<dyn std::error::Error>>;

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
  f: MeshRef,
  fox: MeshRef,
  texture: WebGLTexture,
  angle_x: f64,
  angle_y: f64,
  camera: Camera,
  last_aspect: f32,
  movement: Vector3<f32>,
}

impl State for GameState {
  fn new(ctx: &mut Ctx) -> Result<Self> {
    let vertices = VertexPosTex {
      pos: get_geometry(),
      tex: get_texcoords(),
    };

    let img = load_image(include_bytes!("../static/f-texture.png"))?;

//    let model = ::gltf::Gltf::from_slice(include_bytes!("../static/fox.glb")).unwrap();
//    let images = model.images().collect::<Vec<_>>();
//    let buffers = model.buffers().collect::<Vec<_>>();
//    let document = model.document;
    let (document, buffers, images) = ::gltf::import_slice(include_bytes!("../static/fox.glb").as_ref()).unwrap();

    fn read_buffer<T: Copy>(buffers: &Vec<::gltf::buffer::Data>, accessor: &::gltf::Accessor) -> Vec<T> {
      fn cast<T: Copy>(chunk: &[u8]) -> T {
        unsafe { *(chunk.as_ptr() as *const T) }
      }

      let view = accessor.view();
      let buffer = view.buffer();

      let offset = view.offset();
      let length = view.length();

      let buffer_slice = &buffers[buffer.index()].0[offset..offset + length];

      let data_type = accessor.data_type();

      use gltf::accessor::DataType::*;

      let type_size = match data_type {
        I8 => 1,
        U8 => 1,
        I16 => 2,
        U16 => 2,
        U32 => 4,
        F32 => 4,
      };

      buffer_slice.chunks(type_size).map(|chunk| cast(chunk)).collect::<Vec<_>>()
    }

    let mesh: ::gltf::Mesh = document.meshes().next().unwrap();
    let prim: ::gltf::Primitive = mesh.primitives().next().unwrap();

    let position_accessor = prim.get(&::gltf::Semantic::Positions).unwrap();
    let index_accessor = prim.indices().unwrap();

    let pos_buffer = read_buffer::<f32>(&buffers, &position_accessor);
    let index_buffer = read_buffer::<u16>(&buffers, &index_accessor);

//    log(format!("{:#?}\n{:#?}\n{:#?}", pos_buffer.len() / 3, index_buffer.len() / 3, index_buffer));

    let mut col_buffer = vec![0u8; pos_buffer.len() / 3 * 4];

    let mut rng = rand::thread_rng();
    rng.fill_bytes(&mut col_buffer);

    let fox_vertices = VertexPosCol {
      pos: pos_buffer,
      col: col_buffer,
    };
    let fox_mesh = ctx.renderer().create_mesh(Box::new(fox_vertices), Some(index_buffer));

    let aspect = ctx.renderer().aspect();

    Ok(GameState {
      f: ctx.renderer().create_mesh(Box::new(vertices), None),
      fox: fox_mesh,
      texture: ctx.renderer().create_texture(img.2.as_slice(), img.0, img.1),
      angle_x: 0.0,
      angle_y: 0.0,
      camera: Camera::perspective(Deg(70.0), aspect, 1.0, 2000.0, Point3::new(0.0, 0.0, 0.0)),
      last_aspect: aspect,
      movement: Vector3::new(0.,0.,0.),
    })
  }

  fn update(&mut self, delta: f64, ctx: &Ctx) -> Result<()> {
    let mouse_down = ctx.input().is_down(Button::LeftMouse);
    let (dx, dy) = ctx.input().mouse_delta();

//    log(format!("{:?}", ctx.input().mouse_delta()));

    let sensitivity = 0.1;

    if mouse_down {
      self.angle_x = (self.angle_x + (dx * sensitivity)) % 360.0;
      self.angle_y = (self.angle_y + (dy * sensitivity)) % 360.0;
    }

//    log(format!("({:.2}, {:.2})", self.angle_x, self.angle_y));

    let right = ctx.input().is_down(Button::D);
    let left = ctx.input().is_down(Button::A);
    let forward = ctx.input().is_down(Button::W);
    let back = ctx.input().is_down(Button::S);
    let up = ctx.input().is_down(Button::E);
    let down = ctx.input().is_down(Button::Q);
    let shift = ctx.input().is_down(Button::Shift);

    let movement = Vector3::new(
      if left && !right { 1.0 } else if right && !left { -1.0 } else { 0.0 },
      if up && !down { -1.0 } else if down && !up { 1.0 } else { 0.0 },
      if forward && !back { 1.0 } else if back && !forward { -1.0 } else { 0.0 },
    );

    let rot = Quaternion::from(
      Euler::new(
        Deg(self.angle_y as f32),
        Deg(self.angle_x as f32),
        Deg(0.0)
      )
    );
    let cam_rotation: Matrix4<f32> = Matrix4::from(rot);

    let movement = rot.invert().rotate_vector(movement * 60.0 * delta as f32);
    let cam_pos = Matrix4::from_translation(movement).transform_point(self.camera.get_pos());

    self.camera.set_pos(cam_pos);
    self.camera.set_view(Matrix4::from(rot) * Matrix4::from_translation(self.camera.get_pos().to_homogeneous().truncate()));

    Ok(())
  }

  fn render(&mut self, renderer: &mut Renderer) -> Result<()> {
    {
      let aspect = renderer.aspect();
      if self.last_aspect != aspect {
        let view = self.camera.get_view();
        self.camera = Camera::perspective(Deg(70.0), aspect, 1.0, 2000.0, self.camera.get_pos());
        self.camera.set_view(view);
        self.last_aspect = aspect;
      }
    }
    renderer.clear(0.0, 0.0, 0.0, 0.0);

    let num_fs = 5;
    let radius = 200.0;

    renderer.set_projection(self.camera.combined());

    renderer.render_mesh(self.fox, Matrix4::from_scale(5.0));

    renderer.gl.bind_texture(&self.texture);

    for i in 0..num_fs {
      let angle = i as f32 * ::std::f32::consts::PI * 2.0 / num_fs as f32;

      let x = angle.cos() * radius;
      let z = angle.sin() * radius;

      let transform = Matrix4::from_translation(Vector3::new(x, 0.0, z));

      renderer.render_mesh(self.f, transform);
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
//    js! {@(no_return)
//      console.error(@{format!("{:#?}", info)});
//    }
  }));

  run::<GameState>("Test");
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
    30.0,  0.0,  0.0,
    30.0,  30.0, 0.0,
    100.0, 0.0,  0.0,
    30.0,  30.0, 0.0,
    100.0, 30.0, 0.0,
    100.0, 0.0,  0.0,
    // middle rung front
    30.0, 60.0, 0.0,
    30.0, 90.0, 0.0,
    67.0, 60.0, 0.0,
    30.0, 90.0, 0.0,
    67.0, 90.0, 0.0,
    67.0, 60.0, 0.0,
    // left column back
    0.0,  0.0,   30.0,
    30.0, 0.0,   30.0,
    0.0,  150.0, 30.0,
    0.0,  150.0, 30.0,
    30.0, 0.0,   30.0,
    30.0, 150.0, 30.0,
    // top rung back
    30.0, 0.0, 30.0,
    100.0, 0.0, 30.0,
    30.0, 30.0, 30.0,
    30.0, 30.0, 30.0,
    100.0, 0.0, 30.0,
    100.0, 30.0, 30.0,
    // middle rung back
    30.0, 60.0, 30.0,
    67.0, 60.0, 30.0,
    30.0, 90.0, 30.0,
    30.0, 90.0, 30.0,
    67.0, 60.0, 30.0,
    67.0, 90.0, 30.0,
    // top
    0.0, 0.0, 0.0,
    100.0, 0.0, 0.0,
    100.0, 0.0, 30.0,
    0.0, 0.0, 0.0,
    100.0, 0.0, 30.0,
    0.0, 0.0, 30.0,
    // top rung right
    100.0, 0.0, 0.0,
    100.0, 30.0, 0.0,
    100.0, 30.0, 30.0,
    100.0, 0.0, 0.0,
    100.0, 30.0, 30.0,
    100.0, 0.0, 30.0,
    // under top rung
    30.0, 30.0, 0.0,
    30.0, 30.0, 30.0,
    100.0, 30.0, 30.0,
    30.0, 30.0, 0.0,
    100.0, 30.0, 30.0,
    100.0, 30.0, 0.0,
    // between top rung and middle
    30.0, 30.0, 0.0,
    30.0, 60.0, 30.0,
    30.0, 30.0, 30.0,
    30.0, 30.0, 0.0,
    30.0, 60.0, 0.0,
    30.0, 60.0, 30.0,
    // top of middle rung
    30.0, 60.0, 0.0,
    67.0, 60.0, 30.0,
    30.0, 60.0, 30.0,
    30.0, 60.0, 0.0,
    67.0, 60.0, 0.0,
    67.0, 60.0, 30.0,
    // right of middle rung
    67.0, 60.0, 0.0,
    67.0, 90.0, 30.0,
    67.0, 60.0, 30.0,
    67.0, 60.0, 0.0,
    67.0, 90.0, 0.0,
    67.0, 90.0, 30.0,
    // bottom of middle rung.
    30.0, 90.0, 0.0,
    30.0, 90.0, 30.0,
    67.0, 90.0, 30.0,
    30.0, 90.0, 0.0,
    67.0, 90.0, 30.0,
    67.0, 90.0, 0.0,
    // right of bottom
    30.0, 90.0, 0.0,
    30.0, 150.0, 30.0,
    30.0, 90.0, 30.0,
    30.0, 90.0, 0.0,
    30.0, 150.0, 0.0,
    30.0, 150.0, 30.0,
    // bottom
    0.0, 150.0, 0.0,
    0.0, 150.0, 30.0,
    30.0, 150.0, 30.0,
    0.0, 150.0, 0.0,
    30.0, 150.0, 30.0,
    30.0, 150.0, 0.0,
    // left side
    0.0, 0.0, 0.0,
    0.0, 0.0, 30.0,
    0.0, 150.0, 30.0,
    0.0, 0.0, 0.0,
    0.0, 150.0, 30.0,
    0.0, 150.0, 0.0,
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
