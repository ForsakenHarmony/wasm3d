extern crate webgl_generator;

use webgl_generator::*;
use std::env;
use std::fs::File;
use std::path::*;

fn main() {
  let dest = env::var("OUT_DIR").unwrap();

//  let mut file1 = File::create(&Path::new(&dest).join("test_webgl_stdweb.rs")).unwrap();
//  Registry::new(Api::WebGl, Exts::ALL)
//      .write_bindings(StdwebGenerator, &mut file1)
//      .unwrap();

  let mut file2 = File::create(&Path::new(&dest).join("test_webgl2_stdweb.rs")).unwrap();
  Registry::new(Api::WebGl2, Exts::NONE)
      .write_bindings(StdwebGenerator, &mut file2)
      .unwrap();
}