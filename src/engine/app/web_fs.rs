use std;
use std::cell::RefCell;
use std::io::ErrorKind;
use std::rc::Rc;
use std::str;
use stdweb::web::TypedArray;

pub type IoError = std::io::Error;

pub struct FileSystem {}

enum BufferState {
  Empty,
  Buffer(Vec<u8>),
  Error(String),
}

pub struct File {
  buffer_state: Rc<RefCell<BufferState>>,
}

impl FileSystem {
  pub fn open(file: &str) -> Result<File, IoError> {
    let buffer_state = Rc::new(RefCell::new(BufferState::Empty));

    let on_get_buffer = {
      let buffer_state = buffer_state.clone();
      move |ab: TypedArray<u8>| {
        let data = ab.to_vec();
        if data.len() > 0 {
          *buffer_state.borrow_mut() = BufferState::Buffer(data);
        }
      }
    };

    let on_error = {
      let buffer_state = buffer_state.clone();
      move |s: String| {
//        let msg = format!("Fail to read file from web {}", s);
        *buffer_state.borrow_mut() = BufferState::Error(s);
      }
    };

    js! {@(no_return)
      const filename = @{file};
      fetch(filename)
        .then(r => r.arrayBuffer())
        .then(buffer => {
          const on_get_buffer = @{on_get_buffer};
          on_get_buffer(new Uint8Array(buffer));
          on_get_buffer.drop();
        })
        .catch(e => {
          console.error(e);

          const on_error = @{on_error};
          on_error("Failed to fetch from network");
          on_error.drop();
        });
    }

    Ok(File { buffer_state })
  }
}

impl File {
  pub fn is_ready(&self) -> bool {
    let bs = self.buffer_state.borrow();
    match *bs {
      BufferState::Empty => false,
      BufferState::Error(_) => true,
      BufferState::Buffer(_) => true,
    }
  }

  pub fn read_binary(&mut self) -> Result<Vec<u8>, IoError> {
    let mut bs = self.buffer_state.borrow_mut();
    match *bs {
      BufferState::Error(ref s) => Err(std::io::Error::new(ErrorKind::Other, s.clone())),
      BufferState::Buffer(ref mut v) => Ok({
        let mut r = Vec::new();
        r.append(v);
        r
      }),
      _ => unreachable!(),
    }
  }

  pub fn read_text(&mut self) -> Result<String, IoError> {
    let mut bs = self.buffer_state.borrow_mut();
    match *bs {
      BufferState::Error(ref s) => Err(std::io::Error::new(ErrorKind::Other, s.clone())),
      BufferState::Buffer(ref mut v) => match str::from_utf8(v) {
        Err(e) => Err(std::io::Error::new(ErrorKind::Other, e)),
        Ok(v) => Ok(v.to_owned()),
      },
      _ => unreachable!(),
    }
  }
}
