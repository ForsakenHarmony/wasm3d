use super::AppConfig;

use stdweb::traits::IEvent;
use stdweb::unstable::TryInto;
use stdweb::web::event::{IKeyboardEvent, IMouseEvent, KeyDownEvent, KeyUpEvent, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ResizeEvent, PointerMoveEvent};
use stdweb::web::html_element::CanvasElement;
use stdweb::web::window;
use stdweb::web::IEventTarget;
use stdweb::web::IHtmlElement;

use std::cell::RefCell;
use std::rc::Rc;

use super::AppEvent;

pub struct App {
  window: CanvasElement,
  pub events: Rc<RefCell<Vec<AppEvent>>>,
  device_pixel_ratio: f32,
}

use super::events;

macro_rules! map_events {
  (|$event:ident: $typ:ident| { ($events:expr) < $out:expr }) => {{
    let events = $events.clone();
    move |$event: $typ| {
      events.borrow_mut().push($out);
    }
  }};

  (prevent |$event:ident: $typ:ident| { ($events:expr) < $out:expr }) => {{
    let events = $events.clone();
    move |$event: $typ| {
      $event.prevent_default();
      events.borrow_mut().push($out);
    }
  }};
}

// In browser request full screen can only called under event handler.
// So basically this function is useless at this moment.
#[allow(dead_code)]
fn request_full_screen(canvas: &CanvasElement) {
  js! {
      var c = @{&canvas};
      if (c.requestFullscreen) {
          c.requestFullscreen();
      } else if (c.webkitRequestFullscreen) {
          c.webkitRequestFullscreen(Element.ALLOW_KEYBOARD_INPUT);
      } else if (c.mozRequestFullScreen) {
          c.mozRequestFullScreen();
      } else if (c.msRequestFullscreen) {
          c.msRequestFullscreen();
      }
  };
}

impl App {
  pub fn new(config: AppConfig) -> App {
    use stdweb::web::*;

    if config.headless {
      // Right now we do not support headless in web.
      unimplemented!();
    }

    let canvas: CanvasElement = document()
      .create_element("canvas")
      .unwrap()
      .try_into()
      .unwrap();

    js! {@(no_return)
      const canvas = @{&canvas};
      // setup the buffer size
      // see https://webglfundamentals.org/webgl/lessons/webgl-resizing-the-canvas.html
//      const realToCSSPixels = window.devicePixelRatio;
      canvas.width = 10;
      canvas.height = 10;

      // setup the canvas size
//      canvas.style.width = width + "px";
//      canvas.style.height = height + "px";

      // Make it focusable
      // https://stackoverflow.com/questions/12886286/addeventlistener-for-keydown-on-canvas
      canvas.tabIndex = 1;

      // infinite turning
      canvas.addEventListener("mousedown", ev => {
        canvas.requestPointerLock();
      }, false);
      canvas.addEventListener("mouseup", ev => {
        document.exitPointerLock();
      }, false);

      function resize() {
        const realToCSSPixels = window.devicePixelRatio;

        // Lookup the size the browser is displaying the canvas in CSS pixels
        // and compute a size needed to make our drawingbuffer match it in
        // device pixels.
        const displayWidth  = Math.floor(canvas.clientWidth  * realToCSSPixels);
        const displayHeight = Math.floor(canvas.clientHeight * realToCSSPixels);

        // Check if the canvas is not the same size.
        if (canvas.width  !== displayWidth ||
            canvas.height !== displayHeight) {

          // Make the canvas the same size
          canvas.width  = displayWidth;
          canvas.height = displayHeight;
        }
      }
      window.addEventListener("resize", ev => {
        resize();
      }, false);
      resize();
      Promise.resolve().then(resize);
    };

    if !config.show_cursor {
      js! {@(no_return)
        @{&canvas}.style.cursor="none";
      };
    }

    let device_pixel_ratio: f64 = js! { return window.devicePixelRatio; }.try_into().unwrap();

    let body = document().query_selector("body").unwrap().unwrap();

    body.append_child(&canvas);
    js! {@(no_return)
      @{&canvas}.focus();
    }

    if config.fullscreen {
      println!("Webgl do not support with_screen.");
    }

    let mut app = App {
      window: canvas,
      events: Rc::new(RefCell::new(vec![])),
      device_pixel_ratio: device_pixel_ratio as f32,
    };
    app.setup_listener();

    app
  }

  fn setup_listener(&mut self) {
    let canvas: &CanvasElement = self.canvas();

    canvas.add_event_listener(map_events! {
      |e: MouseDownEvent| {
        (self.events) < AppEvent::MouseDown(
          events::MouseButtonEvent {button:match e.button() {
            MouseButton::Left => 0,
            MouseButton::Wheel => 1,
            MouseButton::Right => 2,
            MouseButton::Button4 => 3,
            MouseButton::Button5 => 4,
          }}
        )
      }
    });
    canvas.add_event_listener(map_events! {
      |e: MouseUpEvent| {
        (self.events) < AppEvent::MouseUp(
          events::MouseButtonEvent {button:match e.button() {
            MouseButton::Left => 0,
            MouseButton::Wheel => 1,
            MouseButton::Right => 2,
            MouseButton::Button4 => 3,
            MouseButton::Button5 => 4,
          }}
        )
      }
    });

    canvas.add_event_listener(map_events! {
      |e: PointerMoveEvent| {
        (self.events) < AppEvent::MousePos(
          (e.client_x() as f64, e.client_y() as f64, e.movement_x() as f64, e.movement_y() as f64), e.buttons()
        )
      }
    });

    canvas.add_event_listener(map_events! {
      prevent |e: KeyDownEvent| {
        (self.events) < AppEvent::KeyDown(
          events::KeyDownEvent {
            code: e.code(),
            key: e.key(),
            shift: e.shift_key(),
            alt: e.alt_key(),
            ctrl: e.ctrl_key(),
          }
        )
      }
    });

    // canvas.add_event_listener(map_event!{
    //     self.events,
    //     KeypressEvent,
    //     KeyPress,
    //     e,
    //     events::KeyPressEvent {
    //         code: e.code()
    //     }
    // });

    canvas.add_event_listener(map_events! {
      prevent |e: KeyUpEvent| {
        (self.events) < AppEvent::KeyUp(
          events::KeyUpEvent {
            code: e.code(),
            key: e.key(),
            shift: e.shift_key(),
            alt: e.alt_key(),
            ctrl: e.ctrl_key(),
          }
        )
      }
    });

    window().add_event_listener({
      let canvas = canvas.clone();

      map_events! {
        |e: ResizeEvent| {
          (self.events) < AppEvent::Resized((canvas.offset_width() as u32, canvas.offset_height() as u32))
        }
      }
    });
  }

  pub fn print<T: Into<String>>(msg: T) {
    js! { console.log(@{msg.into()})};
  }

  pub fn get_params() -> Vec<String> {
    let params = js! { return window.location.search.substring(1).split("&"); };
    params.try_into().unwrap()
  }

  pub fn hidpi_factor(&self) -> f32 {
    return self.device_pixel_ratio;
  }

  pub fn canvas(&self) -> &CanvasElement {
    &self.window
  }

  pub fn run_loop<F>(mut self, mut callback: F)
  where
    F: 'static + FnMut(&mut Self, f64) -> (),
  {
    window().request_animation_frame(move |t: f64| {
      callback(&mut self, t);
      self.events.borrow_mut().clear();
      self.run_loop(callback);
    });
  }

  pub fn poll_events<F>(&mut self, callback: F) -> bool
  where
    F: FnOnce(&mut Self) -> (),
  {
    callback(self);
    self.events.borrow_mut().clear();

    true
  }

  pub fn run<F>(self, callback: F)
  where
    F: 'static + FnMut(&mut Self, f64) -> (),
  {
    self.run_loop(callback);
  }

  pub fn set_fullscreen(&mut self, _b: bool) {
    // unimplemented!();
  }
}

pub fn now() -> f64 {
  // perforamce now is in ms
  // https://developer.mozilla.org/en-US/docs/Web/API/Performance/now
  let v = js! { return performance.now() / 1000.0; };
  return v.try_into().unwrap();
}
