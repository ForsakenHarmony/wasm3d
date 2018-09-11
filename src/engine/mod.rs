#![allow(unused)]

pub mod app;
pub mod webgl;

pub mod mesh;
pub mod renderer;
pub mod shader;

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

use self::mesh::{Mesh, VertexFormat};
use self::renderer::{Camera, Renderer};
use self::shader::ShaderConfig;

type Result<R> = ::std::result::Result<R, Box<::std::error::Error>>;



pub trait State {
  fn new(renderer: &mut Renderer) -> Result<Self> where Self: ::std::marker::Sized;
  fn update(&mut self, delta: f64) -> Result<()>;
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

  let mut start = 0.0;
  let mut last = 0.0;
  let mut frametimes = Vec::new();

  web_app.run(move |app, t| {
    let t = t / 1000.0;
    let delta = t - last;
    last = t;

    let before = self::app::now();

    for event in app.events.borrow_mut().drain(..) {
      state.event(event);
    }

    state.update(delta);

    gl.viewport(0, 0, size.0, size.1);

    renderer.start();

    state.render(&mut renderer);

    let after = self::app::now();
    frametimes.push((after - before) * 1000.0);

    if after - start >= 1.0 {
      let time = after - start;
      let mut min = ::std::f64::MAX;
      let mut max = 0.0;
      let count = frametimes.len() as f64;
//      renderer.gl.log(format!("FPS: {:?})", frametimes));
      let avg = frametimes.drain(..).fold(0.0, |acc, val| {
        if val < min {
          min = val;
        } else if val > max {
          max = val;
        }
        acc + val
      }) / count;
      let fps = count / time;
      start = after;

      renderer.gl.log(format!("FPS: {}\tFrametime: {:.2}ms (min: {:.2}ms, max: {:.2}ms)", fps as u32, avg, min, max));
    };
  });
}
