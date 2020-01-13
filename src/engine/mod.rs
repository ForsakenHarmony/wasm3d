#![allow(unused)]

pub mod app;
pub mod webgl;

pub mod mesh;
pub mod renderer;
pub mod shader;
pub mod input;

use cgmath::{
  perspective, Deg, EuclideanSpace, Matrix, Matrix4, Point3, Rad, SquareMatrix, Transform, Vector3,
};
use crate::engine::app::App as WebApp;
use crate::engine::app::*;
use crate::engine::webgl::*;
use std::collections::HashMap;
use std::panic;
use std::rc::Rc;
use std::any::TypeId;

use crate::engine::mesh::{Mesh, VertexFormat};
use crate::engine::renderer::{Camera, Renderer};
use crate::engine::shader::ShaderConfig;
use crate::engine::app::AppEvent::MouseDown;
use crate::engine::input::Input;

pub type Result<R> = std::result::Result<R, Box<dyn ::std::error::Error>>;

pub trait State {
  fn new(renderer: &mut Ctx) -> Result<Self> where Self: ::std::marker::Sized;
  fn update(&mut self, delta: f64, ctx: &Ctx) -> Result<()>;
  fn render(&mut self, renderer: &mut Renderer) -> Result<()>;
}

pub struct Ctx {
  renderer: Renderer,
  input: Input,
}

impl Ctx {
  pub fn new(renderer: Renderer) -> Self {
    Ctx {
      renderer,
      input: Input::new()
    }
  }

  pub fn renderer(&mut self) -> &mut Renderer {
    &mut self.renderer
  }

  pub fn input(&self) -> &Input {
    &self.input
  }
}

fn log<T: Into<String>>(msg: T) {
  js! {@(no_return)
    console.log(@{msg.into()});
  }
}

pub fn run<T: State>(title: &'static str) where T: 'static {
  let config = AppConfig::new(title);
  let web_app = WebApp::new(config);
  let mut size = (100, 100);

  log(format!("size: {:?})", size));

  let gl = WebGL2RenderingContext::new(web_app.canvas());
  let gl = Rc::new(gl);

  let aspect = size.0 as f32 / size.1 as f32;

  let mut renderer = Renderer::new(Rc::clone(&gl), size, ShaderConfig::default());
  let mut ctx = Ctx::new(renderer);

  let mut state = T::new(&mut ctx).unwrap();

  let mut start = 0.0;
  let mut last = 0.0;
  let mut frametimes = Vec::new();

  let mut first = true;

  web_app.run(move |app, t| {
    if first {
      first = false;
      size = (app.canvas().width(), app.canvas().height());
    }

    let t = t / 1000.0;
    let delta = t - last;
    last = t;

    let before = self::app::now();

    ctx.input.flush();
    for event in app.events.borrow_mut().drain(..) {
      match event {
        AppEvent::Resized(new_size) => {
          log(format!("size: {:?})", new_size));
          size = new_size;
          ctx.renderer().set_size(size);
        },
        _ => {},
      }
      ctx.input.handle_event(event);
    }

    state.update(delta, &ctx);

    gl.viewport(0, 0, size.0, size.1);

    ctx.renderer().start();

    state.render(ctx.renderer());

    ctx.renderer().exec();

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

//      ctx.renderer().gl.log(format!("FPS: {}\tFrametime: {:.2}ms (min: {:.2}ms, max: {:.2}ms)", fps as u32, avg, min, max));
    };
  });
}
