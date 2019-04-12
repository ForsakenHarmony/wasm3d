use crate::engine::app::AppEvent;
use stdweb::web::event::{MouseButtonsState, MouseButton};
use std::collections::HashMap;
use crate::log;
use core::borrow::BorrowMut;
use crate::engine::app::web_app::App;
use crate::engine::app::events::MouseButtonEvent;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Button {
  /// The left mouse button.
  LeftMouse,
  /// The mouse wheel/middle mouse button.
  MiddleMouse,
  /// The right mouse button.
  RightMouse,
  /// The fourth mouse button (browser back).
  Mouse4,
  /// The fifth mouse button (browser forward).
  Mouse5,

  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
  L,
  M,
  N,
  O,
  P,
  Q,
  R,
  S,
  T,
  U,
  V,
  W,
  X,
  Y,
  Z,
}

impl Button {
  pub fn from_code(code: String) -> Option<Self> {
//    log(code.clone());
    Some(match code.as_ref() {
      "KeyA" => Button::A,
      "KeyB" => Button::B,
      "KeyC" => Button::C,
      "KeyD" => Button::D,
      "KeyE" => Button::E,
      "KeyF" => Button::F,
      "KeyG" => Button::G,
      "KeyH" => Button::H,
      "KeyI" => Button::I,
      "KeyJ" => Button::J,
      "KeyK" => Button::K,
      "KeyL" => Button::L,
      "KeyM" => Button::M,
      "KeyN" => Button::N,
      "KeyO" => Button::O,
      "KeyP" => Button::P,
      "KeyQ" => Button::Q,
      "KeyR" => Button::R,
      "KeyS" => Button::S,
      "KeyT" => Button::T,
      "KeyU" => Button::U,
      "KeyV" => Button::V,
      "KeyW" => Button::W,
      "KeyX" => Button::X,
      "KeyY" => Button::Y,
      "KeyZ" => Button::Z,
      _ => return None,
    })
  }
}

#[derive(Clone, Debug)]
pub struct Input {
  /// `x, y, dx, dy`
  mouse: (f64, f64, f64, f64),
  // `state, has_changed`
  map: HashMap<Button, (bool, bool)>,
  size: (u32, u32),
}

impl Input {
  pub fn new() -> Self {
    Input {
      mouse: (0.0, 0.0, 0.0, 0.0),
      map: HashMap::new(),
      size: (0, 0),
    }
  }

  pub fn mouse_delta(&self) -> (f64, f64) {
    (self.mouse.2, self.mouse.3)
  }

  pub fn mouse_pos(&self) -> (f64, f64) {
    (self.mouse.0, self.mouse.1)
  }

  pub fn is_down(&self, button: Button) -> bool {
    self.map.get(&button).map_or(false, |e| e.0)
  }

  pub fn is_pressed(&self, button: Button) -> bool {
    self.map.get(&button).map_or(false, |e| e.1)
  }

  pub(crate) fn flush(&mut self) {
    self.mouse.2 = 0.0;
    self.mouse.3 = 0.0;
    for v in self.map.values_mut() {
      v.1 = false;
    }
  }

  pub(crate) fn handle_event(&mut self, event: AppEvent) -> Option<()> {
    match event {
      AppEvent::MouseDown(e) => {
        let b = self.map.entry(match e.button {
          0 => Button::LeftMouse,
          1 => Button::MiddleMouse,
          2 => Button::RightMouse,
          3 => Button::Mouse4,
          4 => Button::Mouse5,
          _ => return None,
        }).or_insert((false, false));
        b.1 = !b.0;
        b.0 = true;
      }
      AppEvent::MouseUp(e) => {
        let b = self.map.entry(match e.button {
          0 => Button::LeftMouse,
          1 => Button::MiddleMouse,
          2 => Button::RightMouse,
          3 => Button::Mouse4,
          4 => Button::Mouse5,
          _ => return None,
        }).or_insert((false, false));
        b.1 = b.0;
        b.0 = false;
      }
      AppEvent::KeyDown(e) => {
        let b = self.map.entry(Button::from_code(e.code)?).or_insert((false, false));
        b.1 = !b.0;
        b.0 = true;
      }
      AppEvent::KeyUp(e) => {
        let b = self.map.entry(Button::from_code(e.code)?).or_insert((false, false));
        b.1 = b.0;
        b.0 = false;
      }
      AppEvent::Resized(size) => {
        self.size = size;
      }
      AppEvent::MousePos(p, state) => {
        self.mouse = p;
        for b in vec![MouseButton::Left, MouseButton::Wheel, MouseButton::Right, MouseButton::Button4, MouseButton::Button5].into_iter() {
          let evt = MouseButtonEvent {
            button: match b {
              MouseButton::Left => 0,
              MouseButton::Wheel => 1,
              MouseButton::Right => 2,
              MouseButton::Button4 => 3,
              MouseButton::Button5 => 4,
            }
          };
          self.handle_event(if state.is_down(b) { AppEvent::MouseDown(evt) } else { AppEvent::MouseUp(evt) });
        }
      }
    };
    Some(())
  }
}
